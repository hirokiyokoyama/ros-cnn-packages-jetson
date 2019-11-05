#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"
xhost +
ros-container -it --name openpose-visualize --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros visualize.py image:=/camera/image_rect_color
