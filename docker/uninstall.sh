#!/bin/bash

# Parameters
SERVICE="ivit-i"

# Stop Service
sudo systemctl stop ${SERVICE}

# Disable Service when startup
sudo systemctl disable ${SERVICE}

# Remove service file
sudo rm /etc/systemd/system/${SERVICE}.service

# Reload Service
sudo systemctl daemon-reload