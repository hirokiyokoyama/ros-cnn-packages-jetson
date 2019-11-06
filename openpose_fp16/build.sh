#!/bin/bash

git clone https://github.com/hirokiyokoyama/openpose_ros.git
docker build -t ros-cnn-packages-jetson-openpose-fp16 .
