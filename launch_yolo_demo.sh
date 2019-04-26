#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=xavier-1.local \
	             --net=host"
xhost +
ros-container -d --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ros-cnn-packages-jetson-yolo \
	      rosrun yolo_ros visualize.py image:=/camera/color/image_rect_color
ros-container -it --rm ros-cnn-packages-jetson-yolo

