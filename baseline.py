import os
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

LOGGER = logging.getLogger(__name__)
FIGURE = plt.figure(figsize=(9, 6), dpi=200)
AX = FIGURE.gca()
IM = None
LOG_PATH = ''
WEIGHT = 'yolov5s'
CACHE_DIR = os.path.expanduser('~/Data/waymo/cache')
IMAGE_FILES = ["segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
               "segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord",
               "segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord",
               "segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord",
               "segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord",
               'segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord',
               'segment-10075870402459732738_1060_000_1080_000_with_camera_labels.tfrecord',
               'segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord',
               'segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord',
               ]
CACHED_FILES = []


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
    half = device.type != 'cpu'
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=device)['model'].float()
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
def process_image(device, model, model_classify, opt, index, data, width, height, timestamp, frame_sequence, log_path):
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
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    if opt.classify:
        pred = apply_classifier(pred, model_classify, img, img0)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(pred):  # detections per image
        if opt.log_detections:
            detection_log_path = os.path.join(log_path, f'{frame_sequence}.{WEIGHT}.txt')
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
                cv2.imwrite(os.path.join(log_path, f"{frame_sequence}.jpg"), img0)
                draw_image(img0)
                plt.draw()
                plt.pause(.01)


def read_images(device, model, model_classify, opt, frame_sequences):
    log_path = os.path.join(opt.path, 'baseline')
    Path(log_path).mkdir(parents=True, exist_ok=True)
    for seq in frame_sequences:
        width, height, timestamp = 1920, 1280, -1
        image = np.load(CACHED_FILES[seq])
        process_image(device, model, model_classify, opt, -1, image, width, height, timestamp, seq, log_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='the path of expeiment dir')
    parser.add_argument('-w', '--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('-v', '--view-img', action='store_true', help='display results')
    parser.add_argument('-d', '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-2', '--classify', default=False, help='whether to enable the second-level classifier')
    parser.add_argument('-s', '--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('-c', '--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('-i', '--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('-g', '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('-a', '--augment', action='store_true', help='augmented inference')
    parser.add_argument('-f', '--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('-l', '--show-images', action='store_true', help='Show detected objects')
    parser.add_argument('-r', '--redo', action='store_true', help='Ignore the finish tag')
    parser.add_argument('--log-detections', default='detection.json', help='The json file to record detected objects')
    opt = parser.parse_args()
    global WEIGHT
    WEIGHT = opt.weights.split('/')[-1].split('.')[0]
    return opt


def object_detection(opt, frame_sequences):
    try:
        device, model, model_classify = get_model(opt)
        read_images(device, model, model_classify, opt, frame_sequences)
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")


def main():
    opt = parse_args()
    for f in IMAGE_FILES:
        f = os.path.join(CACHE_DIR, f)
        ff = sorted([i for i in os.listdir(f) if i.endswith('npy')], key=lambda x: int(x.split('.')[0]))
        CACHED_FILES.extend([os.path.join(f, i) for i in ff])
    frame_sequences = set()
    for f in os.listdir(opt.path):
        if f == 'baseline':
            continue
        f = os.path.join(opt.path, f)
        f = os.path.join(f, 'dump')
        for i in [int(i.split('.')[0]) for i in os.listdir(f) if i.endswith('.bin')]:
            frame_sequences.add(i)
    object_detection(opt, frame_sequences)


SERVER_PROTOCOLS = set()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")
