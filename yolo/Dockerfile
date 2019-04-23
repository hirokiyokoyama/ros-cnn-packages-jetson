FROM ros-cnn-packages-jetson-common
LABEL maintainer "hirokiyokoyama <h-yokoyama@craft-server.co.jp>"

# setup yolo_ros
RUN git clone https://github.com/hirokiyokoyama/yolo_ros.git
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y yolo_ros \
    && rm -rf /var/lib/apt/lists/*

# download data
WORKDIR /catkin_ws/src/yolo_ros/data
RUN gdrive download 1qu5rdCz0DUAr_vdkmfLWxLJgpp1WzExd
RUN tar -zxvf yolo_ros_data.tar.gz && rm -f yolo_ros_data.tar.gz

RUN . /catkin_ws/devel/setup.sh && cd /catkin_ws && catkin_make
CMD ["rosrun", "yolo_ros", "yolo.py", \
     "--ckpt", "/catkin_ws/src/yolo_ros/data/yolo9000.ckpt", \
     "--names", "/catkin_ws/src/yolo_ros/data/9k.names", \
     "--tree", "/catkin_ws/src/yolo_ros/data/9k.tree", \
     "image:=/camera/color/image_rect_color"]
