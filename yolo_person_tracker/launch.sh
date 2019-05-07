#!/bin/bash

docker build -t ros-cnn-packages-jetson-yolo-person-tracker .

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=xavier-1.local \
	             --net=host"
xhost +
ros-container -d --name yolo-person-tracker-yolo --rm ros-cnn-packages-jetson-yolo-person-tracker
ros-container -it --name yolo-person-tracker-person-tracker --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
              -v `pwd`/scripts:/catkin_ws/src/yolo_ros/person_tracker_scripts \
	      ros-cnn-packages-jetson-yolo-person-tracker \
	      rosrun yolo_ros person_tracker.py image:=/camera/color/image_rect_color
