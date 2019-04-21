FROM hirokiyokoyama/catkin_ws
LABEL maintainer "hirokiyokoyama <hiroki.yokoyama@okadanet.org>"

# setup yolo_ros
RUN git clone https://github.com/hirokiyokoyama/openpose_ros.git
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y openpose_ros \
    && rm -rf /var/lib/apt/lists/*

# download data
WORKDIR /catkin_ws/src/openpose_ros/data
RUN gdrive download 1JSMwXyygLuARtJfsAue_BBs7k-AfuB15
RUN tar -zxvf openpose_ros_data.tar.gz && rm -f openpose_ros_data.tar.gz

RUN . /catkin_ws/devel/setup.sh && cd /catkin_ws && catkin_make

