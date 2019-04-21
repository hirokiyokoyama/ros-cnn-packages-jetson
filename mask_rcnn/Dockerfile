FROM hirokiyokoyama/catkin_ws
LABEL maintainer "hirokiyokoyama <hiroki.yokoyama@okadanet.org>"

# setup mask_rcnn_ros
RUN git clone https://github.com/hirokiyokoyama/mask_rcnn_ros.git
RUN . /catkin_ws/devel/setup.sh && apt-get update && rosdep update \
    && rosdep install -y mask_rcnn \
    && rm -rf /var/lib/apt/lists/*

# download data
WORKDIR /catkin_ws/src/mask_rcnn_ros/data
RUN gdrive download 1_PM6sEsqcktr-HOZ7uV6BJRIqH_gBPlv
RUN tar -zxvf mask_rcnn_ros_data.tar.gz && rm -f mask_rcnn_ros_data.tar.gz

RUN . /catkin_ws/devel/setup.sh && cd /catkin_ws && catkin_make
RUN cd /catkin_ws/src/mask_rcnn_ros/src/mask_rcnn && make

