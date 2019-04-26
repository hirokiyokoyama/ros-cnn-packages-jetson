FROM ros-cnn-packages-jetson-common
LABEL maintainer "hirokiyokoyama <h-yokoyama@craft-server.co.jp>"

# setup yolo_ros
RUN git clone https://github.com/hirokiyokoyama/yolo_ros.git
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y yolo_ros \
    && rm -rf /var/lib/apt/lists/*

# download data
WORKDIR /catkin_ws/src/yolo_ros/data
ADD ./tiny-yolo-voc.tar.gz .

# to update source files
RUN cd /catkin_ws/src/yolo_ros && git pull && git checkout ebd649e

RUN . /opt/ros/melodic/setup.sh && . /catkin_ws/devel/setup.sh \
    && cd /catkin_ws && catkin_make
CMD ["rosrun", "yolo_ros", "yolo.py", \
     "--ckpt", "/catkin_ws/src/yolo_ros/data/tiny-yolo-voc.ckpt", \
     "--names", "/catkin_ws/src/yolo_ros/data/voc.names", \
     "image:=/camera/color/image_rect_color"]
