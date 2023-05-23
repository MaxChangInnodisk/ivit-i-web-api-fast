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
QUICK=false
RUN_SERVICE=true

# Help
function help(){
	echo "Run the iVIT-I environment."
	echo
	echo "Syntax: scriptTemplate [-bcpqh]"
	echo "options:"
	echo "b		Run in background."
	echo "c		Run command line mode."
	echo "q		Qucik start."
	echo "h		help."
}


# Get information from argument
while getopts "bcqh:" option; do
	case $option in
		b )
			INTERATIVE=false ;;
		c )
			RUN_SERVICE=false ;;
		q )
			QUICK=true ;;
		h )
			help; exit ;;
		\? )
			help; exit ;;
		* )
			help; exit ;;
	esac
done

# ========================================================
# Initialize Docker Command Option

# [NAME]
DOCKER_IMAGE="${DOCKER_USER}/${PROJECT}-${PLATFORM}:${VERSION}-${TAG}"
DOCKER_NAME="${PROJECT}-${PLATFORM}-${VERSION}-${TAG}"

# [BASIC]
WS="/workspace"
SET_NAME="--name ${DOCKER_NAME}"
MOUNT_WS="-w ${WS} -v $(pwd):${WS}"
SET_TIME="-v /etc/localtime:/etc/localtime:ro"
SET_NETS="--net=host"

# [DEFINE COMMAND]
# [DEFINE COMMAND]
RUN_CMD=""
CLI_CMD="bash"
WEB_CMD="python3 main.py"

# [DEFINE OPTION]
SET_CONTAINER_MODE=""
SET_VISION=""
SET_PRIVILEG=""
SET_MEM=""
MOUNT_ACCELERATOR=""

# ========================================================

# [ACCELERATOR]
SET_PRIVILEG="--privileged -v /dev:/dev"
SET_MEM="--ipc=host"
MOUNT_ACCELERATOR="\
-v /sys:/sys -v /etc/vart.conf:/etc/vart.conf -v /run:/run -v /lib/firmware:/lib/firmware "
MOUNT_TOOL_FOR_TEMPRATURE="\
-v /usr/bin/xmutil:/usr/bin/xmutil -v /usr/bin/platformstats:/usr/bin/platformstats \
-v /usr/lib/libplatformstats.so.1:/usr/lib/libplatformstats.so.1 "
MOUNT_X264="\
-v $(pwd)/docker/patch/libgstx264.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstx264.so"


# ========================================================
# [VISION] Set up Vision option for docker if need
if [[ ! -z $(echo ${DISPLAY}) ]];then
	SET_VISION="-v /tmp/.x11-unix:/tmp/.x11-unix:rw -e DISPLAY=unix${DISPLAY}"
	xhost + > /dev/null 2>&1
	printd " * Detected monitor" R
fi

# ========================================================
# [Basckground] Update background option
if [[ ${INTERATIVE} = true ]]; then 
	SET_CONTAINER_MODE="-it"
	printd " * Run Interative Terminal Mode" R
else
	SET_CONTAINER_MODE="-dt"; 
	printd " * Run Background Mode" R
fi

# Checking Run CLI or Web
if [[ ${RUN_SERVICE} = true ]]; then 
	RUN_CMD="${RUN_CMD} ${WEB_CMD}"
	printd " * Run Web API Directly" R
else 
	RUN_CMD="${RUN_CMD} ${CLI_CMD}"
	printd " * Run Command Line Interface" R
fi

# Conbine docker command line
DOCKER_CMD="docker run \
--rm \
${SET_CONTAINER_MODE} \
${SET_NAME} \
${SET_PRIVILEG} \
${MOUNT_ACCELERATOR} \
${MOUNT_TOOL_FOR_TEMPRATURE} \
${MOUNT_X264} \
${SET_NETS} \
${SET_MEM} \
${SET_TIME} \
${MOUNT_WS} \
${SET_VISION} \
${DOCKER_IMAGE} ${RUN_CMD}"

# ========================================================
# Logout and wait
echo -ne "\n${DOCKER_CMD}\n"
echo ""
if [[ ${QUICK} = false ]];then waitTime 5; fi

# ========================================================
# Execution

# Run Docker Service
if [[ $(docker &>/dev/null) -eq 1 ]];then
	sudo dockerd & > ./docker.log
fi

# Rund Docker Compose
printd "Launch Relative Container" G
docker compose --file ${DOCKER_COMPOSE} -p ${TAG} up -d 

# Run docker command 
printd "Launch iVIT-I Container" G
bash -c "${DOCKER_CMD}"

if [[ ${INTERATIVE} = true ]];then
	printd "Close Relative Container" R
	docker compose -f ${DOCKER_COMPOSE} -p ${TAG} down
fi

exit 0;
