#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
		    -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"

ros-container -it --name openpose-openpose --rm \
	      ros-cnn-packages-jetson-openpose-fp16 \
              rosrun openpose_ros openpose_fp16.py \
	      image:=/camera/image_rect_color
