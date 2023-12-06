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
DOCKER_USER="innodiskorg"
DOCKER_COMPOSE="docker/compose.yml"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Parse information from configuration
check_jq
PROJECT=$(cat "${CONF}" | jq -r '.PROJECT')
VERSION=$(cat "${CONF}" | jq -r '.VERSION')
PLATFORM=$(cat "${CONF}" | jq -r '.PLATFORM')
TAG=$(cat "${CONF}" | jq -r '.TAG')

# ========================================================
# [NAME]
DOCKER_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"
DOCKER_NAME="${PROJECT}-${PLATFORM}-${VERSION}-${TAG}"

# ========================================================
# Stop
COMPOSE_CMD=$(get_docker_compose)
${COMPOSE_CMD} -f ${DOCKER_COMPOSE} -p ${TAG} down || echo "Stop relatived service failed."

docker stop ${DOCKER_NAME} || echo "Stop iVIT-I service failed."
