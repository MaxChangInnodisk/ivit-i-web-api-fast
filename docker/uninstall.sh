#!/bin/bash

# Parameters
SERVICE="ivit-i"

# Disable Service when startup
sudo systemctl disable ${SERVICE}

# Remove service file
sudo rm /etc/systemd/system/${SERVICE}.service

# Reload Service
sudo systemctl daemon-reload