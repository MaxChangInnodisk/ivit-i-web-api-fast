#!/bin/bash
ROOT=$(pwd)
apt-get install -yq libx264-dev git ninja-build
pip3 install meson
git clone -b 1.18 https://github.com/GStreamer/gst-plugins-ugly.git
cd gst-plugins-ugly
meson build
cd build
ninja
cp ./ext/x264/libgstx264.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
cp ./ext/x264/libgstx264.so ${ROOT}/docker/patch

# cd ${ROOT}
# rm -rf gst-plugins-ugly