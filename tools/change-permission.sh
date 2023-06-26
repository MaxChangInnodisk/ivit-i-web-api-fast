#!/bin/bash

sudo chown 1000:1000 -R . 2> /dev/null || echo "No need root"
chown 1000:1000 -R .
