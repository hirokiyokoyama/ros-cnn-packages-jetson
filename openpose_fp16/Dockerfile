FROM ros-cnn-packages-jetson-common
LABEL maintainer "hirokiyokoyama <h-yokoyama@craft-server.co.jp>"

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
 && apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup scipy
RUN apt-get update && apt-get install -y \
    gfortran liblapack-dev unzip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && pip3 install scipy

# setup ros
RUN apt-get update && apt-get install -y \
    ros-melodic-image-geometry \
    && rm -rf /var/lib/apt/lists/*

# setup openpose_ros
ADD openpose_ros /catkin_ws/src/openpose_ros
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y openpose_ros \
    && rm -rf /var/lib/apt/lists/*
#RUN . /catkin_ws/devel/setup.sh && cd /catkin_ws && catkin_make
RUN . /opt/ros/melodic/setup.sh && cd /catkin_ws && catkin_make

# download data
WORKDIR /catkin_ws/src/openpose_ros/data
RUN gdrive download 1NAR2NZ4Lb1aSlGXEq2mALRdT5FtadBME
RUN unzip openpose_fp16.zip && rm -f openpose_fp16.zip

CMD ["rosrun", "openpose_ros", "openpose_fp16.py", \
     "image:=/camera/color/image_rect_color"]

