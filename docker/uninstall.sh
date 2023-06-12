#!/bin/bash

# Parameters
SERVICE="ivit-i"

# Disable Service when startup
sudo systemctl disable ${SERVICE}
