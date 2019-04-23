FROM hirokiyokoyama/ros-jp42:py3.6-tf1.13.1
LABEL maintainer "hirokiyokoyama <h-yokoyama@craft-server.co.jp>"

COPY ./gdrive /usr/local/bin/gdrive
RUN mkdir /root/.gdrive
COPY ./token_v2.json /root/.gdrive
