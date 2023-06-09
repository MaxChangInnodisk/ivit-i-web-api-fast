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
DOCKER_USER="maxchanginnodisk"
DOCKERFILE="docker/ivit-i-service.dockerfile"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Parse information from configuration
check_jq
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