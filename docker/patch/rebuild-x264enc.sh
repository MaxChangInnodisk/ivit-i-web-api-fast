#!/bin/bash
ROOT=$(pwd)
PROJECT="gst-plugins-ugly"

# Install relatived package
apt-get install -yq libx264-dev git ninja-build
pip3 install meson

# Remove legacy project if it exists
rm -rf ${PROJECT} || echo "Gst Plugin Ugly not exists"

# Clone 1.18 gst-plugin-ugly
git clone -b 1.18 https://github.com/GStreamer/gst-plugins-ugly.git
cd ${PROJECT}

# Rebuild
meson build
cd build
ninja

if [[ -f ./ext/x264/libgstx264.so ]];then
    ls ./ext/x264
    echo "Build x264enc successed!"
    cp ./ext/x264/libgstx264.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0
    cp ./ext/x264/libgstx264.so ${ROOT}/docker/patch
    cd ${ROOT}
    rm -rf gst-plugins-ugly
else
    echo "Build x264enc failed!"
fi

