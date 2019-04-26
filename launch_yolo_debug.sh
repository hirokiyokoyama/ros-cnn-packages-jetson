#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=xavier-1.local \
	             --net=host"
ros-container -it --name yolo-debug ros-cnn-packages-jetson-yolo bash

