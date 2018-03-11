#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = 'ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
#sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2


__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')

    return parser

def predict(net, img, label_colours, input_shape):
    input_image = cv2.imread(img, 1)
    input_image = input_image.astype(np.float32)

    input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray([input_image])

    out = net.forward_all(**{net.inputs[0]: input_image})

    prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)

    prediction = np.squeeze(prediction)
    prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
    prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

    prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
    label_colours_bgr = label_colours[..., ::-1]
    cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
    return prediction_rgb

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape
    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
    cnt = 1
    while cnt < 10000:
      img = '%s/%05d.jpg' % (args.input_image, cnt)
      print(img)
      prediction_rgb = predict(net, img, label_colours, input_shape)
      img = cv2.imread(img, 1)
      h,w,_=prediction_rgb.shape
      img = cv2.resize(img, (w,h))
      cv2.bitwise_and(prediction_rgb,img,prediction_rgb)
      cv2.imshow("ENet", prediction_rgb)
      cnt += 1
      key = cv2.waitKey(1)







