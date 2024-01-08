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
# Basic Parameters
CONF="ivit-i.json"
DOCKER_USER="innodiskorg"
DOCKERFILE="docker/ivit-i-service.dockerfile"
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
        echo "Launching iVIT-I-JETSON"
        ;;
    *)
        echo "Not detect platform !!!!"
        echo "Usage     : build.sh [PLATFORM]"
        echo "Example   : build.sh intel"
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

# Parse Information
PROJECT=$(cat "${CONF}" | jq -r '.PROJECT')
VERSION=$(cat "${CONF}" | jq -r '.VERSION')
IVIT_VERSION=$(cat "${CONF}" | jq -r '.IVIT_VERSION')
PLATFORM=$(cat "${CONF}" | jq -r '.PLATFORM')
TAG=$(cat "${CONF}" | jq -r '.TAG')

# ========================================================
# Execution

# Concatenate Name
SRC_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${IVIT_VERSION}-runtime"
TRG_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"

# Build the docker image
printd "Build the docker image. (${TRG_IMAGE})" Cy

docker build \
-f ${DOCKERFILE} \
--build-arg BASE=${SRC_IMAGE} \
-t "${TRG_IMAGE}" .

# ========================================================
# Testing
docker run -it --rm \
-e DISPLAY=unix:0 "${TRG_IMAGE}" echo "Hello World!"