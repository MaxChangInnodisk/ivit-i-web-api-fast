#!/bin/bash

# Determine the system architecture
ARCH=$(uname -m)
BASEDIR=$(dirname $0)

if [ "$ARCH" == "x86_64" ]; then
    JQ_BINARY="${BASEDIR}/jq-linux-amd64"
elif [ "$ARCH" == "aarch64" ]; then
    JQ_BINARY="${BASEDIR}/jq-linux-arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Make the jq binary executable
chmod +x $JQ_BINARY

# Move the jq binary to a location in the PATH and rename it to 'jq'
cp $JQ_BINARY /usr/bin/jq

# Verify the installation
if [ -f /usr/bin/jq ]; then
    echo "jq has been installed successfully."
else
    echo "Failed to install jq."
fi