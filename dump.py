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
from pathlib import Path


LOGGER = logging.getLogger(__name__)
PORT = 4400
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


def read_shared_mem(opt):
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
                    print(f'New frame is ready, skip frame #{index}[{frame_sequence}], captured at {timestamp} [{time.monotonic() * 1000}]')
                    index += 1
                    continue
            if finished == 1:  # check the finished tag
                bin_path = os.path.join(opt.output, f'{frame_sequence}.bin')
                print(f"write {bin_path}")
                json_path = os.path.join(opt.output, f'{frame_sequence}.json')
                json.dump({'timestamp': timestamp, 'width': width, 'height': height}, open(json_path, 'w+'))
                with open(bin_path, 'wb+') as f:
                    f.write(bytes(shm.buf[CONTENT_OFFSET + offset: CONTENT_OFFSET + offset + length]))
                    f.close()
                index += 1
    shm.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='dump', help='output folder')  # output folder
    opt = parser.parse_args()
    return opt


def dump_images(opt):
    read_shared_mem(opt)


def main():
    opt = parse_args()
    Path(opt.output).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(opt.output):
        f = os.path.join(opt.output, f)
        if os.path.isfile(f):
            os.remove(f)
    dump_images(opt)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        LOGGER.info("Keyboard interruption detected, terminate programme.")

