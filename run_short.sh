#!/usr/bin/env bash

source ~/WORK/environment/poni-multi-clustering/bin/activate

#python3 ~/WORK/EM-MNIST/mnist_em.py mnist_test_30.csv &
python3 ~/WORK/EM-MNIST/gm.py mnist_test_30.csv &

wait %1 