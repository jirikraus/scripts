#!/bin/bash

# Number of gpus with compute_capability 3.5  per server
NGPUS=`nvidia-smi -L | wc -l`

MPS_PIPE_BASE_DIR=${HOME}/tmp/mps/`hostname`/pipe
MPS_LOG_BASE_DIR=${HOME}/tmp/mps/`hostname`/log

# Stop the MPS control daemon for each GPU
echo "Stopping all nvidia-cuda-mps-control daemons:"
for ((i=0; i< ${NGPUS}; i++)); do
	export CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_BASE_DIR}/$i
	echo "quit" | nvidia-cuda-mps-control
	sleep 1
	echo " - stopped nvidia-cuda-mps-control for GPU $i on `hostname`"
done
sleep 5
sudo nvidia-smi -c 0
echo "Done on `hostname`"
