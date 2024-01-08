#!/bin/bash
# Store the utilities
FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")
source "${ROOT}/utils.sh"

if [ $# -ne 1 ]; then
    printd "Usage: load.sh <path/to/tar>" R; exit 1
fi

# Verify Tar file
TAR="$1"
if [ -f "$TAR" ]; then
    printd "Find $TAR"
else
    printd "Can not find $TAR" R; exit 1
fi

# Extract to temperory folder
TMP_FOLDER=$(mktemp -d)
tar -xf "$TAR" -C "$TMP_FOLDER"

# Get all tar file
ALL_TAR=$(find "$TMP_FOLDER" -type f -name "*.tar" -exec realpath {} \;)

# Parse each tar file
for TMP in ${ALL_TAR}; do
    
    # Load docker image
    docker load -i ${TMP} 1>/dev/null
    
    # Logout
    BASENAME="$(basename ${TMP})"
    if [ $? -eq 0 ]; then
        printd "${BASENAME}... PASS !" G
    else
        printd "${BASENAME}... FAIL !" R
    fi
done

# Clear
rm -rf $TMP_FOLDER
if [[ ! -d ${TMP_FOLDER} ]]; then
    printd "Clear temporary folder."
else
    printd "Fail to clear temporary fodler." R
fi
