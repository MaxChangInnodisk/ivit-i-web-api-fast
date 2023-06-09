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

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Parse information from configuration
check_jq
# PLATFORM=$(cat "${CONF}" | jq -r '.PLATFORM')
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
TEMP=".temp"

# ========================================================
# Check configuration is exit
check_config ${CONF}

# ========================================================
# Split Platform and Other option
check_jq

# Array
OPT_ARR=(${*})
if [[ ${#OPT_ARR[@]} -eq 1 ]];then
    echo "Not detect platform !!!!"
    echo "Usage     : run.sh [PLATFORM] [OPTION]"
    echo "Example   : run.sh intel -q"
    exit
fi

# Get platform
PLATFORM=${OPT_ARR[0]}

unset OPT_ARR[0]
OPTS=${OPT_ARR[@]}

jq --arg a "${PLATFORM}" '.PLATFORM = $a' ${CONF} > ${TEMP} && mv ${TEMP} ${CONF} 

# ========================================================
git submodule update --init

# ========================================================
# Switcher
./docker/run-${PLATFORM}.sh ${OPTS}