#!/bin/bash

for i in {1..1000};
do
    python3 nas_bench_201_dataset.py --dataset cifar100 --cuda_num 0
done
