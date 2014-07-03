#!/bin/bash

# Number of gpus with compute_capability 3.5  per server
NGPUS=`nvidia-smi -L | wc -l`

MPS_PIPE_BASE_DIR=${HOME}/tmp/mps/`hostname`/pipe
MPS_LOG_BASE_DIR=${HOME}/tmp/mps/`hostname`/log

#export CUDA_DEVICE_MAX_CONNECTIONS=20
sudo nvidia-smi -c 3
sleep 5
echo "Starting one nvidia-cuda-mps-control daemon for each GPU:"
# Start the MPS server for each GPU
for ((i=0; i< ${NGPUS}; i++)); do
	export CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_BASE_DIR}/$i
	export CUDA_MPS_LOG_DIRECTORY=${MPS_LOG_BASE_DIR}/$i
	mkdir -p ${CUDA_MPS_PIPE_DIRECTORY}
	mkdir -p ${CUDA_MPS_LOG_DIRECTORY}
	export CUDA_VISIBLE_DEVICES=$i
	nvidia-cuda-mps-control -d
	sleep 1
	echo " - started nvidia-cuda-mps-control for GPU $i on `hostname`"
done
echo "Done on `hostname`"
