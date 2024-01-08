#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# ========================================================
# Global Varaible
CONF="ivit-i.json"
DOCKER_USER="innodiskorg"
REL_PATH=$( realpath "../ivit-i-intel")
TAG="service"
TEMP=".temp"

# Define Basic Argument Parameters
PKG_SO=false
CP_FILE=false
BUILD_DOCKER=false

# ========================================================
# Store the utilities
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
source "${ROOT}/utils.sh"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Split Platform and Other option
check_jq

# ========================================================
# Check platform
PLATFORM=$1
case ${PLATFORM} in
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
        echo "Usage     : push_docker_image.sh [PLATFORM]"
        echo "Example   : push_docker_image.sh intel"
        exit
        ;;

esac

PROJECT=$(cat ${CONF} | jq -r '.PROJECT')
VERSION=$(cat ${CONF} | jq -r '.VERSION')
# PLATFORM=$(cat ${CONF} | jq -r '.PLATFORM')
VERSION=$(cat ${CONF} | jq -r '.VERSION')

# Concate name
TRG_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"
CNTR_NAME="${PROJECT}-${PLATFORM}-${VERSION}"

printd "Pushing Docker Image to Docker Hub ... ${TRG_IMAGE}" BR
docker push ${TRG_IMAGE}