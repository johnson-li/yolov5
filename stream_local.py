import argparse
import struct
import socket
import json
import asyncio
import logging
import numpy as np

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from pprint import pprint
from random import randint

LOGGER = logging.getLogger(__name__)
FIGURE = plt.figure(figsize=(9, 6), dpi=200)
AX = FIGURE.gca()
IM = None
LOG_PATH = ''
WEIGHT = 'yolov5s'


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
    weights = opt.weights
    if not weights.endswith('.pt'):
        weights = f'{weights}.pt'
    if not weights.startswith('weights/'):
        weights = f'weights/{weights}'
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()
    model.to(device).eval()
    if half:
        model.half()
    model_classify = torch_utils.load_classifier(name='resnet101', n=2) if opt.classify else None
    if model_classify:
        model_classify.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
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
    # print(
    #     f"Process image #{index}[{frame_sequence}] of size ({width}x{height}) captured at {timestamp} [{time.monotonic() * 1000}]")
    img_size = check_img_size(width)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    img0 = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))  # BGRA
    img = letterbox(img0[:, :, :3], new_shape=img_size)[0]
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
    if opt.log_detections:
        with open(LOG_PATH, 'a+') as f:
            f.write(f'YOLOv5 cost {(end_ts - start_ts) * 1000 :.02f} ms for frame #{frame_sequence}\n')
    # print(f"YOLOv5 took {(end_ts - start_ts) * 1000:.02f} ms, finished at [{time.monotonic() * 1000}]")
    if opt.classify:
        pred = apply_classifier(pred, model_classify, img, img0)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(pred):  # detections per image
        if opt.log_detections:
            detection_log_path = os.path.join(opt.path, f'{frame_sequence}.{WEIGHT}.txt')
            if os.path.exists(detection_log_path):
                os.remove(detection_log_path)
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                if opt.show_images:
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                if opt.log_detections:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    x1 = xywh[0] - xywh[2] / 2
                    y1 = xywh[1] - xywh[3] / 2
                    x2 = xywh[0] + xywh[2] / 2
                    y2 = xywh[1] + xywh[3] / 2
                    result = {'frame_sequence': frame_sequence, 'frame_timestamp': timestamp,
                              'yolo_timestamp': time.clock_gettime(time.CLOCK_MONOTONIC),
                              'detection': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                            'conf': conf.item(), 'cls_conf': conf.item(), 'cls_pred': int(cls),
                                            'cls_pred_name': names[int(cls)]},
                              'yolo_version': 'v5'}
                    line = json.dumps(result) + '\n'
                    if opt.log_detections:
                        with open(detection_log_path, 'a+') as f:
                            f.write(line)
            if opt.show_images:
                cv2.imwrite(os.path.join(opt.path, f"{frame_sequence}.jpg"), img0)
                draw_image(img0)
                plt.draw()
                plt.pause(.01)
        else:
            result = {}
            line = json.dumps(result) + '\n'
            if opt.log_detections:
                with open(detection_log_path, 'a+') as f:
                    f.write(line)



def read_images(device, model, model_classify, opt):
    files = os.listdir(opt.path)
    sequences = sorted([int(f.split('.')[0]) for f in files if f.endswith('.bin')])
    for seq in sequences:
        try:
            meta = json.load(open(os.path.join(opt.path, f'{seq}.json')))
        except Exception as e:
            print('Failed to parse json file: ' + str(os.path.join(opt.path, f'{seq}.json')))
            continue
        width, height, timestamp = meta['width'], meta['height'], meta['timestamp']
        # if width != opt.img_size:
        #     print(f'Image of wrong size: {width}x{height} vs {opt.img_size}')
        #     continue
        image = np.fromfile(os.path.join(opt.path, f'{seq}.bin'))
        process_image(device, model, model_classify, opt, -1, image, width, height, timestamp, seq)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='the path of dump dir')
    parser.add_argument('-w', '--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('-v', '--view-img', action='store_true', help='display results')
    parser.add_argument('-d', '--device', default=str(randint(0,1)), help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-2', '--classify', default=False, help='whether to enable the second-level classifier')
    parser.add_argument('-s', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('-i', '--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('-g', '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('-a', '--augment', action='store_true', help='augmented inference')
    parser.add_argument('-f', '--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('-l', '--show-images', action='store_true', help='Show detected objects')
    parser.add_argument('-r', '--redo', action='store_true', help='Ignore the finish tag')
    parser.add_argument('--log-detections', default='detection.json', help='The json file to record detected objects')
    opt = parser.parse_args()

    #meta = {}
    #with open(os.path.join(opt.path, '../metadata.txt')) as f:
    #    for line in f.readlines():
    #        line = line.strip()
    #        if line:
    #            line = line.split('=')
    #            meta[line[0]] = line[1]
    #opt.img_size = check_img_size(int(meta['resolution'].split('x')[0]))
    #print(f'img_size: {opt.img_size}')
    global WEIGHT
    WEIGHT = opt.weights.split('/')[-1].split('.')[0]
    return opt


def object_detection(opt):
    try:
        device, model, model_classify = get_model(opt)
        read_images(device, model, model_classify, opt)
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")


def main():
    opt = parse_args()
    finish_log = os.path.join(opt.path, f'stream_local.{WEIGHT}.finish')
    if not opt.redo and os.path.isfile(finish_log):
        logging.warning('Already processed, skip')
        return
    if len(os.listdir(opt.path)) < 5:
        logging.warning('Not enough data, skip')
        return
    if opt.log_detections:
        global LOG_PATH
        LOG_PATH = os.path.join(opt.path, f'stream_local.{WEIGHT}.log')
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
    logging.warning(f"Working on {opt.path}")
    object_detection(opt)
    with open(finish_log, 'w+') as f:
        f.write('finished')


SERVER_PROTOCOLS = set()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")
