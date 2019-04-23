#!/bin/bash

docker build -t ros-cnn-packages-jetson-common common

docker build --no-cache -t ros-cnn-packages-jetson-yolo yolo
#docker run --name=ros-cnn-packages-jetson-yolo-tmp ros-cnn-packages-jetson-yolo
#docker commit ros-cnn-packages-jetson-yolo-tmp ros-cnn-packages-jetson-yolo
#docker rm ros-cnn-packages-jetson-yolo-tmp

docker build --no-cache -t ros-cnn-packages-jetson-openpose openpose

