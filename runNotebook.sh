#!/usr/bin/env bash
#enable job control
set -m
jupyter notebook --ip="*" 2> /tmp/url.txt &
sleep 1
grep http /tmp/url.txt | grep -vi notebook | sed "s/localhost/$(hostname)/" > ./url2.txt
fg 



