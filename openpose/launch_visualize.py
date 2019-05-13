#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
	            -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"
xhost +
ros-container -it --name openpose-visualize --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros __visualize.py image:=/camera/color/image_rect_color
