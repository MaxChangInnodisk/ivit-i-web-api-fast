#!/bin/bash

# Variable
PLATFORM=${1}
MODE=${2}

# Parameters
SERVICE="ivit-i"
SERV_FILE=""
CLI_SERV="ivit-i-cli.service"
GUI_SERV="ivit-i-gui.service"
EXEC_CMD=""
OPTS="-q"

# Double Check
if [[ -z ${PLATFORM} || ${PLATFORM} = "-h" || ${PLATFORM} = "--help" ]]; then
	echo "Please setup platform ..."
	echo ""
	echo "Usage:  install.sh [PLATFORM] [MODE]."
	echo ""
	echo "	- PLATFORM: support intel, xilinx, nvidia, jetson, hailo."
	echo "	- MODE: support cli, gui. default is gui."
	exit
fi

# Helper
function update_service_file() {
	root=$2
	file=$1
	start_cmd="$(realpath ${root})/docker/run.sh ${PLATFORM} ${OPTS}"
	stop_cmd="$(realpath ${root})/docker/stop.sh"

	sed -i 's#ExecStart=.*#ExecStart='"${start_cmd}"'#g' $file
	sed -i 's#ExecStop=.*#ExecStop='"${stop_cmd}"'#g' $file
	
}


# Store the utilities
FILE=$(realpath "$0")
DOCKER_ROOT=$(dirname "${FILE}")
ROOT=$(dirname "${DOCKER_ROOT}")
source "${DOCKER_ROOT}/utils.sh"

# Make sure submodule is downlaod
git submodule update --init || echo "Already initailized."

# Select .service file ( cli, gui )
TITLE=""
if [[ ${MODE} = "cli" ]];then
    TITLE="CLI Mode"
    SERV_FILE="$(realpath ${DOCKER_ROOT})/service/${CLI_SERV}"
else
    TITLE="GUI Mode"
	SERV_FILE="$(realpath ${DOCKER_ROOT})/service/${GUI_SERV}"
fi

echo "[${TITLE}] Using service file: ${SERV_FILE}"

# Modify .service file
update_service_file ${SERV_FILE} "${ROOT}" 

# Change Permission
sudo chmod 644 ${SERV_FILE}

# Move to /etc/systemd/system
cp ${SERV_FILE} /etc/systemd/system/${SERVICE}.service

# Reload Service
sudo systemctl daemon-reload

# Start Service
sudo systemctl start ${SERVICE}
# systemctl status ${SERVICE}

# Enable Service when startup
sudo systemctl enable ${SERVICE}
