#!/usr/bin/env bash

# 
# Launch script to run jupyter notebook and record the remote access url in a text file.
# useful for running the notebook on a remote workstation with dropbox syncing of the filesystem.
#

#enable job control
set -m
jupyter notebook --ip="*" 2> /tmp/url.txt &
sleep 1
grep http /tmp/url.txt | grep -vi notebook | sed "s/localhost/$(hostname)/" > ./url2.txt
fg 



