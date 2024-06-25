#!/bin/bash

# Check if jq is installed
if [ -f /usr/bin/jq ]; then
    # Remove the jq binary
    rm /usr/bin/jq

    # Verify the uninstallation
    if [ ! -f /usr/bin/jq ]; then
        echo "jq has been uninstalled successfully."
    else
        echo "Failed to uninstall jq."
    fi
else
    echo "jq is not installed."
fi
