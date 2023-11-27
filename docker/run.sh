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
COMPOSE="${ROOT}/compose.yml"
TEMP=".temp"

# ========================================================
# Check configuration is exit
check_config ${CONF}

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
        echo "Usage     : run.sh [PLATFORM] [OPTION]"
        echo "Example   : run.sh intel -q"
        exit
        ;;

esac

# ========================================================
# Get platform
PLATFORM=${OPT_ARR[0]}

# Clear platform parameter
unset OPT_ARR[0]
OPTS=${OPT_ARR[@]}

# Modify Configuration
jq --arg a "${PLATFORM}" '.PLATFORM = $a' ${CONF} > ${TEMP} && mv -f ${TEMP} ${CONF} 

# ========================================================
# Disclaimer
${ROOT}/disclaim/disclaimer.sh

if [ $? -eq 1 ];then 
    echo "Quit."; exit 0; 
fi

# Update Docker Compose
# ========================================================
# Split Platform and Other option
check_jq
API_PORT=$(cat "${CONF}" | jq -r '.SERVICE.PORT')
NGINX_PORT=$(cat "${CONF}" | jq -r '.NGINX.PORT')
WEB_PORT=$(cat "${CONF}" | jq -r '.WEB.PORT')


update_compose_env ${COMPOSE} "NG_PORT=${NGINX_PORT}"
update_compose_env ${COMPOSE} "API_PORT=${API_PORT}" 

update_compose_env ${COMPOSE} "BACKEND_PORT=${NGINX_PORT}"
update_compose_env ${COMPOSE} "NGINX_PORT=${WEB_PORT}"

# ========================================================
# Download submodule
git submodule update --init || echo "Already initailized."

# ========================================================
# Malcoln Camera Service
./docker/start_camera.sh > /dev/null 2>&1 &
printd "Started camera service ..." BR

# ========================================================
# Switcher
./docker/run-${PLATFORM}.sh ${OPTS}

# ========================================================
# End of Malcoln Camera Service
./docker/end_camera.sh
printd "Stopped camera service ..." BR