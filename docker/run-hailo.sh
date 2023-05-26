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
DOCKER_COMPOSE="./docker/docker-compose.yml"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Parse information from configuration
check_jq
PROJECT=$(cat ${CONF} | jq -r '.PROJECT')
VERSION=$(cat ${CONF} | jq -r '.VERSION')
PLATFORM=$(cat ${CONF} | jq -r '.PLATFORM')
TAG=$(cat "${CONF}" | jq -r '.TAG')

# ========================================================
# Get Option

INTERATIVE=true
RUN_SERVICE=true
QUICK=false

#HAILO docker args 
HAILORT_ENABLE_SERVICE=false #-e hailort_enable_service=yes 
DISABLE_MULTIPLEXER=false #-e HAILO_DISABLE_MULTIPLEXER=1 
ENABLE_MULTI_DEVICE_SCHEDULER=false #-e HAILO_ENABLE_MULTI_DEVICE_SCHEDULER=1 
ENABLE_HAILO_MONITOR=false #-e HAILO_MONITOR=1 
HAILORT_LOGGER_PATH=false #-e HAILORT_LOGGER_PATH=${HAILORT_LOGGER_PATH} 

# Help
function help(){
	echo "Run the iVIT-I environment."
	echo
	echo "Syntax: scriptTemplate [-bcpqh]"
	echo "options:"
	echo "b		Run in background."
	echo "c		Run command line mode."
	echo "p		Select a platform to run ( the priority is higher than ivit-i.json ). support in [ 'intel', 'xilinx' ]"
	echo "q		Qucik start."
	echo "h		help."
}

# Get information from argument
while getopts "bcqp:h" option; do
	case $option in
		b )
			INTERATIVE=false ;;
		c )
			RUN_SERVICE=false ;;
		q )
			QUICK=true ;;
		p )
			PLATFORM=${OPTARG} ;;
		h )
			help; exit ;;
		\? )
			help; exit ;;
	esac
done

# ========================================================
# Initialize Docker Command Variables

# [NAME]
DOCKER_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"
DOCKER_NAME="${PROJECT}-${PLATFORM}-${VERSION}-${TAG}"

# [BASIC]
WS="/workspace"
SET_NAME="--name ${DOCKER_NAME}"
MOUNT_WS="-w ${WS} -v $(pwd):${WS}"
SET_TIME="-v /etc/localtime:/etc/localtime:ro"
SET_NETS="--net=host"

# [Hailo]
FIREWARE="-v /lib/firmware:/lib/firmware"
KERNEL_MODULE="-v /lib/modules:/lib/modules"
UDEV="-v /lib/udev/rules.d:/lib/udev/rules.d"

# [DEFINE COMMAND]
RUN_CMD=""
CLI_CMD="bash"
WEB_CMD="python3 main.py"

# [PLACEHOLDER]
SET_CONTAINER_MODE="-it"
SET_VISION=""
SET_PRIVILEG="--privileged"
MOUNT_CAM="-v /dev:/dev"
SET_MEM="--ipc=host"

# ========================================================

# Checking Run CLI or Web
if [[ ${RUN_SERVICE} = true ]]; then 
	RUN_CMD="${RUN_CMD} ${WEB_CMD}"
	printd " * Run Web API Directly" R
else 
	RUN_CMD="${RUN_CMD} ${CLI_CMD}"
	printd " * Run Command Line Interface" R
fi


# [ACCELERATOR]
MOUNT_ACCELERATOR="--device /dev/dri --device-cgroup-rule='c 189:* rmw'"

# [VISION] Set up Vision option for docker if need
if [[ ! -z $(echo ${DISPLAY}) ]];then
	SET_VISION="-v /tmp/.x11-unix:/tmp/.x11-unix:rw -e DISPLAY=unix${DISPLAY}"
	xhost + > /dev/null 2>&1
	printd " * Detected monitor"
else
	printd " * Can not detect monitor"
fi

# [Basckground] Update background option
if [[ ${INTERATIVE} = true ]]; then 
	printd " * Run Interative Terminal Mode"
else
	SET_CONTAINER_MODE="-dt"; 
	printd " * Run Background Mode"
fi

# ========================================================
# Conbine docker command line
DOCKER_CMD="docker run \
--rm \
${SET_CONTAINER_MODE} \
${SET_NAME} \
${SET_PRIVILEG} \
${MOUNT_ACCELERATOR} \
${MOUNT_CAM} \
${FIREWARE} \
${KERNEL_MODULE} \
${UDEV} \
${SET_NETS} \
${SET_MEM} \
${SET_TIME} \
${MOUNT_WS} \
${SET_VISION} \
${DOCKER_IMAGE} ${RUN_CMD}"

# ========================================================
# Logout and wait
echo -ne "\n${DOCKER_CMD}\n\n"
if [[ ${QUICK} = false ]];then waitTime 5; fi

# ========================================================
# Execution

# Rund Docker Compose
printd "Launch Relative Container" G
docker compose --file ${DOCKER_COMPOSE} -p ${TAG} up -d 

# Run docker command 
printd "Launch iVIT-I Container" G
docker rm -f ${DOCKER_NAME} &> /dev/null

bash -c "${DOCKER_CMD}"

if [[ ${INTERATIVE} = true ]];then
	printd "Close Relative Container" R
	docker compose -f ${DOCKER_COMPOSE} -p ${TAG} down
fi

exit 0;