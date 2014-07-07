#!/bin/bash

#run script for MPI 

MPS_PIPE_BASE_DIR=${HOME}/tmp/mps/`hostname`/pipe 
MPS_LOG_BASE_DIR=${HOME}/tmp/mps/`hostname`/log 

export CUDA_VISIBLE_DEVICES=0 
if [ ${OMPI_COMM_WORLD_LOCAL_RANK} -lt $(( ${OMPI_COMM_WORLD_LOCAL_SIZE} / 2 )) ]; then 
	export CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_BASE_DIR}/0
	exec $* 
else 
	export CUDA_MPS_PIPE_DIRECTORY=${MPS_PIPE_BASE_DIR}/1 
	exec $* 
fi

