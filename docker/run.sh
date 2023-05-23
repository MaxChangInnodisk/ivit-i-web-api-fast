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
PLATFORM=$(cat "${CONF}" | jq -r '.PLATFORM')

# ========================================================
# Switcher
./docker/run-${PLATFORM}.sh ${*}
