#!/bin/sh
. ./script/address.cfg

sshfs $USER@$IP_ADDRESS:$WORKSTATION_PATH $LOCAL_MOUNT_POINT -p $PORT