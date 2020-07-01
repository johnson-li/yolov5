import argparse
import struct
import socket
import json
import asyncio
import logging

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from multiprocessing import shared_memory, Process, Manager, Barrier

LOGGER = logging.getLogger(__name__)
FRAMES_SIZE = 128
INDEX_SIZE = 24
HEADER_SIZE = 8
BUFFER_SIZE = 100 * 1024 * 1024
CONTENT_OFFSET = 4096
CONTENT_SIZE = BUFFER_SIZE - CONTENT_OFFSET
UNIX_SOCKET_NAME = '/tmp/yolo_stream'
SERVER_BARRIER = Barrier(2)
FIGURE = plt.figure(figsize=(9, 6), dpi=200)
AX = FIGURE.gca()
IM = None


def log_time(name):
    def wrapping(func):
        def wrapped(*args, **kwargs):
            ts = time.time()
            ans = func(*args, **kwargs)
            diff = (time.time() - ts) * 1000  # in millisecond
            print('%s causes %f ms' % (name, diff))
            return ans

        return wrapped

    return wrapping


@log_time("Initiating model")
def get_model(opt):
    device = torch_utils.select_device(opt.device)
    if os.path.exists(opt.output):
        shutil.rmtree(opt.output)  # delete output folder
    os.makedirs(opt.output)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16
    model_classify = torch_utils.load_classifier(name='resnet101', n=2) if opt.classify else None
    if model_classify:
        model_classify.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        model_classify.to(device).eval()
    return device, model, model_classify


def draw_image(img):
    global IM
    if not IM:
        IM = AX.imshow(img)
        plt.show(block=False)
    else:
        IM.set_data(img)


@log_time("Processing image")
def process_image(device, model, model_classify, opt, index, data, width, height, timestamp, frame_sequence):
    print("Process image #%d[%d] of size (%dx%d) captured at %d" % (index, frame_sequence, width, height, timestamp))
    half = device.type != 'cpu'  # half precision only supported on CUDA
    img0 = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))  # BGRA
    img = letterbox(img0[:, :, :3], new_shape=opt.img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, width x height x channel to channel x width x height
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    start_ts = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    end_ts = torch_utils.time_synchronized()
    print("YOLOv5 takes %fms" % ((end_ts - start_ts) * 1000))
    if opt.classify:
        pred = apply_classifier(pred, model_classify, img, img0)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(pred):  # detections per image
        out = opt.output
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                if opt.log_detections:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    result = {'frame_sequence': frame_sequence, 'frame_timestamp': timestamp,
                              'yolo_timestamp': time.clock_gettime(time.CLOCK_MONOTONIC),
                              'detection': {'x1': xyxy[0].item(), 'y1': xyxy[1].item(),
                                            'x2': xyxy[2].item(), 'y2': xyxy[3].item(),
                                            'conf': conf.item(), 'cls_conf': conf.item(), 'cls_pred': int(cls),
                                            'cls_pred_name': names[int(cls)]},
                              'yolo_version': 'v5'}
                    line = json.dumps(result) + '\n'
                    with open(opt.log_detections, 'a+') as f:
                        f.write(line)
                    on_result(result)
            cv2.imwrite(os.path.join(out, "%d.jpg" % frame_sequence), img0)
            if opt.show_images:
                draw_image(img0)
                plt.draw()
                plt.pause(.01)


def read_shared_mem(device, model, model_classify, opt):
    print('Read shared memory')
    while True:
        try:
            shm = shared_memory.SharedMemory(name='/webrtc_frames', create=False)
            break
        except Exception as e:
            print("The shared memory is not ready, sleep 1s")
            time.sleep(1)
    assert shm.size == BUFFER_SIZE

    def get_size():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_offset():
        return struct.unpack('I', bytes(shm.buf[:4]))[0]

    def get_frame_info(i):
        return struct.unpack('IIHHIIi',
                             bytes(shm.buf[HEADER_SIZE + i * INDEX_SIZE: HEADER_SIZE + (i + 1) * INDEX_SIZE]))

    size = get_size()
    print("Number of frames: %d" % size)
    index = size - 1 if size > 0 else 0
    while True:
        if index < get_size():
            i = index % FRAMES_SIZE
            offset, length, width, height, timestamp, frame_sequence, finished = get_frame_info(i)
            if get_size() > index:
                _, _, _, _, next_timestamp, next_frame_sequence, next_finished = get_frame_info((i + 1) % FRAMES_SIZE)
                if next_finished == 1:
                    print('New frame is ready, skip frame #%d[%d], captured at %d' % (
                        index, next_frame_sequence, next_timestamp))
                    index += 1
                    continue
            if finished == 1:  # check the finished tag
                process_image(device, model, model_classify, opt, index,
                              bytes(shm.buf[CONTENT_OFFSET + offset: CONTENT_OFFSET + offset + length]),
                              width, height, timestamp, frame_sequence)
                index += 1
    shm.close()


def on_result(result):
    result = json.dumps(result)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(UNIX_SOCKET_NAME)
    sock.send(result.encode())


class UdpServerProtocol(asyncio.DatagramProtocol):
    def __init__(self):
        self.port = 4400
        self.transport_ = None
        self.clients_ = Manager().dict()

    def connection_made(self, transport):
        self.transport_ = transport

    def connection_lost(self, transport):
        pass

    def on_result(self, result):
        for client in self.clients_.keys():
            print('Send result to: %s, result: %s' % (client, result))
            self.transport_.sendto(result.encode(), client)

    def datagram_received(self, data, addr):
        message = data.decode()
        if addr not in self.clients_:
            print("Register new client: %s" % (addr,))
            self.clients_[addr] = 1


class UnixServerProtocol(asyncio.BaseProtocol):
    def __init__(self):
        self._transport = None
        self._server_protocol = SERVER_PROTOCOL

    def connection_made(self, transport):
        self._transport = transport

    def data_received(self, data):
        data = data.decode().strip()
        self._server_protocol.on_result(data)

    def eof_received(self):
        pass


async def start_udp_server():
    print('Starting UDP server')
    loop = asyncio.get_running_loop()
    transport1, _ = await loop.create_datagram_endpoint(lambda: SERVER_PROTOCOL,
                                                        local_addr=('0.0.0.0', SERVER_PROTOCOL.port))
    unix_server = await loop.create_unix_server(UnixServerProtocol, path=UNIX_SOCKET_NAME)
    try:
        await asyncio.sleep(5)
        SERVER_BARRIER.wait()
        await asyncio.sleep(3600)  # Serve for 1 hour.
    finally:
        transport1.close()
        unix_server.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('-o', '--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('-s', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('-i', '--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('-v', '--view-img', action='store_true', help='display results')
    parser.add_argument('-d', '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-2', '--classify', default=False, help='whether to enable the second-level classifier')
    parser.add_argument('-g', '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('-a', '--augment', action='store_true', help='augmented inference')
    parser.add_argument('-f', '--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('-l', '--show-images', action='store_true', help='Show detected objects')
    parser.add_argument('--log-detections', default='detection.json', help='The json file to record detected objects')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    return opt


def object_detection(opt):
    try:
        device, model, model_classify = get_model(opt)
        SERVER_BARRIER.wait()
        read_shared_mem(device, model, model_classify, opt)
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")


def main():
    opt = parse_args()
    process = Process(target=object_detection, args=(opt,))
    process.start()
    asyncio.run(start_udp_server())
    process.join()


SERVER_PROTOCOL = UdpServerProtocol()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")
