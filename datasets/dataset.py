import argparse
import json
import time

import datautils
from name import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='seed')
    parser.add_argument('--data', type=str, default='MCF-7', help='data')
    parser.add_argument('--trainsz', type=float, default=0.7, help='train size')
    parser.add_argument('--testsz', type=float, default=0.15, help='test size')
    args = parser.parse_args()

    seed = args.seed
    data = args.data
    trainsz = args.trainsz
    testsz = args.testsz

    assert trainsz + testsz < 1

    datautils.set_seed(seed)
    print("Generator info:")
    print(json.dumps(args.__dict__, indent='\t'))

    start = time.time()
    datautils.gen_nodeattr(data)
    datautils.split_train_val_test(data, trainsz, testsz)
    end = time.time()

    print("Generate successfully, time cost: {}".format(end - start))
