#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# ========================================================
# Global Varaible
CONF="ivit-i.json"
DOCKER_USER="maxchanginnodisk"
REL_PATH=$( realpath "../ivit-i-intel")
TAG="service"

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

FLAG=$(ls ${CONF} 2>/dev/null)
if [[ -z $FLAG ]];then 
	printd "Couldn't find configuration (${CONF})" Cy; 
	exit
else 
	printd "Detected configuration (${CONF})" Cy; 
fi

# ========================================================
# Parse information from configuration
check_jq
PROJECT=$(cat ${CONF} | jq -r '.PROJECT')
VERSION=$(cat ${CONF} | jq -r '.VERSION')
PLATFORM=$(cat ${CONF} | jq -r '.PLATFORM')
VERSION=$(cat ${CONF} | jq -r '.VERSION')

# Concate name
TRG_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"
CNTR_NAME="${PROJECT}-${PLATFORM}-${VERSION}"

printd "Pushing Docker Image to Docker Hub ... ${TRG_IMAGE}" BR
docker push ${TRG_IMAGE}