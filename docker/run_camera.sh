#!/bin/bash

export GST_PLUGIN_PATH=/usr/lib/gstreamer-1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
export LIBVA_DRIVER_NAME=iHD
export GST_GL_PLATFORM=egl

sudo modprobe v4l2loopback devices=2

gst-launch-1.0 icamerasrc device-name=ev2m_gom1 ! video/x-raw,format=UYVY,width=1920,height=1080 ! v4l2sink device=/dev/video131 &
gst-launch-1.0 icamerasrc device-name=ev2m_gom1-2 ! video/x-raw,format=UYVY,width=1920,height=1080 ! v4l2sink device=/dev/video132 &

# sudo ./end.sh

# gst-launch-1.0 v4l2src device=/dev/video131 ! videoconvert ! videoscale ! video/x-raw,width=960,height=540 ! videomixer name=mix sink_1::xpos=960 ! videoconvert ! fpsdisplaysink video-sink=ximagesink sync=false v4l2src device=/dev/video132 ! videoconvert ! videoscale ! video/x-raw,width=960,height=540 ! mix.