#!/bin/bash

docker build -t ros-cnn-packages-jetson-openpose .

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=xavier-1.local \
	             --net=host"
xhost +
ros-container -d --name openpose-openpose --rm \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros _openpose.py _stage:=6 image:=/camera/color/image_rect_color
ros-container -it --name openpose-visualize --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros visualize.py image:=/camera/color/image_rect_color
