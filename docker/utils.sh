#!/bin/bash
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# Dear Developer
# this script is link from `tools/utils.sh` to `docker/utils.sh`
# the command I execute is `ln ./tools/utils.sh docker/utilsh`

REST='\e[0m'
GREEN='\e[0;32m'
BGREEN='\e[7;32m'
RED='\e[0;31m'
BRED='\e[7;31m'
YELLOW='\e[0;33m'
BYELLOW='\e[7;33m'
Cyan='\033[0;36m'
BCyan='\033[7;36m'

function printd(){            
    
    if [ -z $2 ];then COLOR=$REST
    elif [ $2 = "G" ];then COLOR=$GREEN
	elif [ $2 = "BG" ];then COLOR=$BGREEN
	elif [ $2 = "R" ];then COLOR=$RED
    elif [ $2 = "BR" ];then COLOR=$BRED
	elif [ $2 = "Y" ];then COLOR=$YELLOW
    elif [ $2 = "BY" ];then COLOR=$BYELLOW
    elif [ $2 = "Cy" ];then COLOR=$Cyan
    elif [ $2 = "BCy" ];then COLOR=$BCyan
    else COLOR=$REST
    fi

    echo -e "$(date +"%y:%m:%d %T") ${COLOR}$1${REST}"
}

function check_image(){ 
	echo "$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep ${1} | wc -l )" 
}
function check_container(){ 
	echo "$(docker ps -a --format "{{.Names}}" | grep ${1} | wc -l )" 
}

function check_container_run(){
	echo "$( docker container inspect -f '{{.State.Running}}' ${1} )"
}

function lower_case(){
	echo "$1" | tr '[:upper:]' '[:lower:]'
}
function upper_case(){
	echo "$1" | tr '[:lower:]' '[:upper:]'
}

function print_magic(){
	info=$1
	magic=$2
	echo ""
	if [[ $magic = true ]];then
		echo -e $info | boxes -d dog -s 80x10
	else
		echo -e $info
	fi
	echo ""
}

function check_jq(){
	# Install pre-requirement
	if [[ -z $(which jq) ]];then
		printd "Installing requirements .... " Cy
		sudo apt-get install jq -yqq
	fi
}

function check_pyinstaller(){
	# Install pyinstaller for inno-verify
	if [[ -z $(which pyinstaller) ]];then
		printd "Installing pyinstaller for inno-verify .... " Cy
		pip3 install setuptools pyinstaller -q
	fi
}

function check_boxes(){
	# Setup Masgic package
	if [[ -z $(which boxes) ]];then 
		printd "Preparing MAGIC "; 
		sudo apt-get install -qy boxes > /dev/null 2>&1; 
	fi
}

function check_lsof(){
	if [[ -z $(which lsof) ]];then
		printd "Preparing lsof "; 
		apt-get install -qy lsof > /dev/null 2>&1; 
	fi
}

function run_webrtc_server(){
	printd "Launch WebRTC to Web Docker Service" Cy
	docker run --rm -d \
	--name ivit-webrtc-server \
	--network host \
	ghcr.io/deepch/rtsptoweb:latest
}

function stop_webrtc_server(){
	docker stop ivit-webrtc-server
}

function update_compose_env() {
	local args=("$@")
	local file=$1

	if [[ $# -lt 2 ]]; then usage; fi
	
	for ((i=1; i<${#args[@]}; i++)); do
		
		local pair=(${args[i]//=/ })
		pair[1]="${pair[1]//\//\\/}"
		sed -Ei "s/(.*${pair[0]}=).*/\1${pair[1]}/g" $file
		echo "Replacing: ${pair[0]} with ${pair[1]}" 

	done
}

function waitTime(){
	TIME_FLAG=$1
	while [ $TIME_FLAG -gt 0 ]; do
		printf "\rWait ... (${TIME_FLAG}) "; sleep 1
		(( TIME_FLAG-- ))
	done
	printf "\r                 \n"
}

function check_folder(){
	TRG_PATH=$1
	if [[ ! -d ${TRG_PATH} ]];then
		mkdir ${TRG_PATH}	
	fi
}

function check_config(){
	CONF=$1
	FLAG=$(ls ${CONF} 2>/dev/null)
	if [[ -z $FLAG ]];then 
		printd "Couldn't find configuration (${CONF})" Cy; 
		exit
	else 
		printd "Detected configuration (${CONF})" Cy; 
	fi
}