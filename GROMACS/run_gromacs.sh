#!/bin/bash
#PBS -N GROMACS-multi
#PBS -l nodes=2:ppn=20
#PBS -q ivb20_k40
#PBS -l walltime=00:10:00
#PBS -m bea
#PBS -j oe
#PBS -M jkraus@nvidia.com

#Setup environment (TODO: Included needed modules)
source /home-2/jkraus/local/gromacs-5.0-rc1-20140507_0432/bin/GMXRC.bash

MDRUN=`which mdrun_mpi`

#MAXH=0.2
MAXH=0.01

#Relies on PBS_NUM_NODES, PBS_NP and PBS_NUM_PPN and assumes
# that PBS_NUM_PPN = number of cores per node
# and that mpiexec is called on a compute node
# OMP_NUM_THREADS=$1
# PBS_NP%OMP_NUM_THREADS == 0
#ARGS
# $1 OMP_NUM_THREADS
# $2 NPME				Number of separate ranks to be used for PME, -1 is guess
# $3 NSTLIST			Set nstlist when using a Verlet buffer tolerance (0 is guess)
# $4 NB					Calculate non-bonded interactions on: auto, cpu, gpu, gpu_cpu
# $5 INPUT				Input tpr file
# $6 LOG_FILE_NAME		name for the GROMACS log file
function launch_gromacs_pbs_openmpi {
	OMP_NUM_THREADS=$1							#Number of OpenMP threads per MPI rank to use
	export OMP_NUM_THREADS
	if [ $(( ${PBS_NP}%${OMP_NUM_THREADS} )) -ne 0 ]; then
		echo "ERROR: ${PBS_NP}%${OMP_NUM_THREADS} != 0 "
		exit 1
	fi 
	NP=$(( ${PBS_NP}/${OMP_NUM_THREADS} ))
	if [ $(( ${NP}%${PBS_NUM_NODES} )) -ne 0 ]; then
		echo "ERROR: ${NP}%${PBS_NUM_NODES} != 0"
		exit 1
	fi 
	PPN=$(( ${NP}/${PBS_NUM_NODES} ))
	
	NUM_SOCKETS=$(( `numactl --show | grep nodebind | tr ' ' '\n' | grep -v nodebind | wc -l` - 1 ))
	NGPUS=`nvidia-smi -L | wc -l`
	
	if [ ${PPN} -eq ${NUM_SOCKETS} ]; then
		MAPPING_POLICY="--bind-to socket --map-by ppr:${PPN}:node "
	elif [ ${PPN} -eq ${PBS_NUM_PPN} ]; then
		MAPPING_POLICY="--bind-to core --map-by core"
	elif [ ${PPN} -eq 1 ]; then
		MAPPING_POLICY="--bind-to none --map-by node"
	else
		MAPPING_POLICY="--bind-to none --map-by ppr:${PPN}:node"
	fi
	
	GPU_ID_CMD=""
	NPPPN=${PPN}
	NPME=$2										#Number of separate ranks to be used for PME, -1 is guess
	if [ "${NB}" != "cpu" ] ; then
	
		if [ ${NPME} -gt 0 ] && [ ${NGPUS} -gt 0 ] ; then
			if [ $(( ${NPME}%${PBS_NUM_NODES} )) -ne 0 ]; then
				echo "ERROR: ${NPME}%${PBS_NUM_NODES} != 0"
				exit 1
			fi
			NPMEPN=$(( ${NPME}/${PBS_NUM_NODES} ))	# Number of PME ranks per node
			NPPPN=$(( ${PPN} - ${NPMEPN} ))			# Number of PP ranks per node
		fi
		
		if [ ${NPPPN} -gt ${NGPUS} ]; then
			temp=0
			GPU_ID_CMD=" -gpu_id `for (( pp=1; pp<=${NPPPN}; pp++ )); do echo ${temp}; temp=$(( (${temp}+1)%${NGPUS} )); done | sort | tr -d '\n'` "
		fi
	
	fi

	NSTLIST=$3 	#Set nstlist when using a Verlet buffer tolerance (0 is guess)
	NB=$4		#Calculate non-bonded interactions on: auto, cpu, gpu, gpu_cpu 
	
	INPUT=$5
		
	if [ $6 ]; then
		LOG_FILE_NAME=$6
	else
		input_basename=$(basename "${INPUT}" .tpr)
		LOG_FILE_NAME="${input_basename}.${NB}.${NP}x${OMP_NUM_THREADS}.md.log"
	fi 
	ENER_FILE_NAME="${LOG_FILE_NAME}.ener"
	
	cmd="mpirun --display-map --report-bindings ${MAPPING_POLICY} -np ${NP} ${MDRUN} -ntomp ${OMP_NUM_THREADS} -nstlist ${NSTLIST} -npme ${NPME} ${GPU_ID_CMD} -nb ${NB} -s ${INPUT} -g ${LOG_FILE_NAME} -e ${ENER_FILE_NAME} -maxh ${MAXH} -resethway -noconfout -v"
	echo "${cmd}"
	${cmd}
}

#Set Boost Clocks for GPUs (Values assume max boost for Tesla K40)
mpirun --map-by ppr:1:node -np $PBS_NUM_NODES nvidia-smi -ac 3004,875

INPUT=/home-2/jkraus/workspace/GROMACS/run/KTH-Cray-RFP/data/ion_channel.tpr

for ompt in 1 5 10 20; do
	# 							OMP_NUM_THREADS NPME NSTLIST NB  INPUT    LOG_FILE_NAME
	launch_gromacs_pbs_openmpi 	${ompt}         0    0       gpu ${INPUT}
	#launch_gromacs_pbs_openmpi ${ompt}         -1   0       cpu ${INPUT}
done

INPUT=/home-2/jkraus/workspace/GROMACS/run/KTH-Cray-RFP/data/lignocellulose.tpr
