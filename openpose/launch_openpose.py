#!/bin/bash

shopt -s expand_aliases
alias ros-container="nvidia-docker run \
	            -e ROS_MASTER_URI=http://tx2-1.local:11311 \
		    -e ROS_HOSTNAME=`hostname`.local \
	            --net=host"

ros-container -it --name openpose-openpose --rm \
              -v `pwd`/scripts:/catkin_ws/src/openpose_ros/_scripts \
	      ros-cnn-packages-jetson-openpose \
	      rosrun openpose_ros openpose.py _stage:=6 __name:=openpose_`hostname`\
	      compute_openpose:=compute_openpose_`hostname`
