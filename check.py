import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='the path of dump dir')
    parser.add_argument('-w', '--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('-r', '--remove', action='store_true',
                        help='remove .finish file when finding missing detection result files')
    parser.add_argument('-c', '--clean', action='store_true', help='delete zero-sized image files')
    opt = parser.parse_args()
    return opt


def main():
    args = parse_args()
    path = args.path
    weight = args.weights
    if not weight.endswith('.pt'):
        weight = weight + '.pt'
    if not weight.startswith('weight/'):
        weight = 'weight/' + weight
    weight_name = weight.split('/')[-1].split('.')[0]
    for p in os.listdir(path):
        if p == 'baseline':
            continue
        p = os.path.join(path, p)
        dump_dir = os.path.join(p, 'dump')
        indexes = [int(i.split('.')[0]) for i in os.listdir(dump_dir) if i.endswith('.bin')]
        indexes = sorted(indexes)
        if len(indexes) < 20:
            print(f'Not enough frames: {p}')
            continue
        for i in indexes:
            bin_path = os.path.join(dump_dir, f'{i}.bin')
            if args.clean and os.path.getsize(bin_path) == 0:
                print(f'Delete zero-sized image: {bin_path}')
                os.remove(bin_path)
                continue
            detection_path = os.path.join(dump_dir, f'{i}.{weight_name}.txt')
            if not os.path.isfile(detection_path):
                print(f'Missing objection detection result file: {detection_path}')
                finish_path = os.path.join(dump_dir, f'stream_local.{weight_name}.finish')
                if args.remove and os.path.isfile(finish_path):
                    print(f'Deleting .finish file: {finish_path}')
                    os.remove(finish_path)
                break


if __name__ == '__main__':
    main()
