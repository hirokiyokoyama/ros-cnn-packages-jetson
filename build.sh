#!/bin/bash

docker build -t ros-cnn-packages-jetson-common common

docker build -t ros-cnn-packages-jetson-yolo yolo
docker run --name=ros-cnn-packages-jetson-yolo-tmp ros-cnn-packages-jetson-yolo
docker commit ros-cnn-packages-jetson-yolo-tmp ros-cnn-packages-jetson-yolo
docker rm ros-cnn-packages-jetson-yolo-tmp

