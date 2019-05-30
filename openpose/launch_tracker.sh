#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
	            -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"
ros-container -it --name openpose-tracker --rm \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros tracker.py \
	      camera_info:=/camera/camera_info
