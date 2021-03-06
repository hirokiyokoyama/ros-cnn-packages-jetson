#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"

ros-container -it --name openpose-openpose --rm \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
              -v `pwd`/data:/catkin_ws/src/openpose_ros/_data \
	      ros-cnn-packages-jetson-openpose \
              rosrun openpose_ros openpose_xavier.py \
	      image:=/camera/image_rect_color
