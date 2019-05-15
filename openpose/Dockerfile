FROM ros-cnn-packages-jetson-common
LABEL maintainer "hirokiyokoyama <h-yokoyama@craft-server.co.jp>"

# setup openpose_ros
RUN git clone https://github.com/hirokiyokoyama/openpose_ros.git
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y openpose_ros \
    && rm -rf /var/lib/apt/lists/*

# setup scipy
RUN apt-get update && apt-get install -y \
    gfortran liblapack-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install scipy

# download data
WORKDIR /catkin_ws/src/openpose_ros/data
RUN gdrive download 1JSMwXyygLuARtJfsAue_BBs7k-AfuB15
RUN tar -zxvf openpose_ros_data.tar.gz && rm -f openpose_ros_data.tar.gz

RUN apt-get update && apt-get install -y ros-melodic-image-geometry \
    && rm -rf /var/lib/apt/lists/*

# update source codes (in case the dependency changes, above must be performed)
RUN cd .. && git pull && git checkout 23d0efb 

#RUN . /catkin_ws/devel/setup.sh && cd /catkin_ws && catkin_make
RUN . /opt/ros/melodic/setup.sh && cd /catkin_ws && catkin_make
CMD ["rosrun", "openpose_ros", "openpose.py", \
     "image:=/camera/color/image_rect_color"]

