#!/bin/bash
python train.py
echo "testing"
python test.py --resume cityscapes/encnet/default/model_best.pth.tar --eval

