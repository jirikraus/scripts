#!/bin/bash

echo "input,nb,npme,nnode,np,nompt,performance(ns/day)"
grep "Performance:" $* | awk '{ gsub(/\./, ",",$1); print}'| awk '{print $1 $2;}' | awk '{ sub(/,md,log:Performance:/, ","); print }' | awk '{ sub(/x/, ","); print }'


