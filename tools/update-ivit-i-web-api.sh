#!/bin/bash

# Store the utilities
FILE=$(realpath "$0")
TOOL_DIR=$(dirname "${FILE}")
ROOT=$(dirname ${TOOL_DIR})
source "${ROOT}/docker/utils.sh"
cd $ROOT


# Parameters
CONF=${ROOT}/ivit-i.json
DOCKER_USER="innodiskorg"

# Update Repo
git fetch && git pull

# Update Submodule
cd apps && git pull && cd ..

# Update Docker image
# Check configuration is exit
check_config ${CONF}
check_jq

PROJECT=$(cat ${CONF} | jq -r '.PROJECT')
VERSION=$(cat ${CONF} | jq -r '.VERSION')
PLATFORM=$(cat ${CONF} | jq -r '.PLATFORM')
VERSION=$(cat ${CONF} | jq -r '.VERSION')
TAG=$(cat ${CONF} | jq -r '.TAG')


# Concate name
TRG_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"

printd "Pulling Docker Image from Docker Hub ... ${TRG_IMAGE}" BR
docker pull ${TRG_IMAGE}