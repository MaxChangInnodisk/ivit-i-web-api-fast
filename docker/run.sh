#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# ========================================================
# Store the utilities
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
source "${ROOT}/utils.sh"

# ========================================================
# Move to correct path
cd $(dirname ${ROOT})

# ========================================================
# Basic Parameters
CONF="ivit-i.json"
TEMP=".temp"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Split Platform and Other option
check_jq

# ========================================================
# Check platform

# Array
OPT_ARR=(${*})

case ${OPT_ARR[0]} in
    "intel" )
        echo "Launching iVIT-I-INTEL"
        ;;
    "xilinx" )
        echo "Launching iVIT-I-XILINX"
        ;;
    "hailo" )
        echo "Launching iVIT-I-HAILO"
        ;;
    "nvidia" )
        echo "Launching iVIT-I-NVIDIA"
        ;;
    "jetson" )
        echo "Launching iVIT-I-NVIDIA"
        ;;
    *)
        echo "Not detect platform !!!!"
        echo "Usage     : run.sh [PLATFORM] [OPTION]"
        echo "Example   : run.sh intel -q"
        exit
        ;;

esac

# Get platform
PLATFORM=${OPT_ARR[0]}

# Clear platform parameter
unset OPT_ARR[0]
OPTS=${OPT_ARR[@]}

# Modify Configuration
jq --arg a "${PLATFORM}" '.PLATFORM = $a' ${CONF} > ${TEMP} && mv -f ${TEMP} ${CONF} 

# ========================================================
# Download submodule
git submodule update --init || echo "Already initailized."

# ========================================================
# Switcher
./docker/run-${PLATFORM}.sh ${OPTS}