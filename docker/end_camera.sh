#!/bin/bash

# Ensure execute the script with the super user

max_retries=10
retry_interval=2
retries=0

# Stop gst launch
pkill -2 gst-launch-1.0

# Stop v4l2loopback
while [ $retries -lt $max_retries ]; do
    
    rmmod v4l2loopback > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        break
    else
        sleep $retry_interval
        retries=$((retries+1))
    fi
done