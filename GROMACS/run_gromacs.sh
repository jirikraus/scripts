#!/bin/bash -
#SBATCH --job-name="gromacs"
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jkraus@nvidia.com
##SBATCH --time=00:15:00
#ntasks-per-core=2 to enable hyper threading
##SBATCH --ntasks-per-core=2
##SBATCH --cpu_bind=no 
##SBATCH --ntasks=8
##SBATCH --cpus-per-task=2
# Submit with: sbatch run_gromacs.sh daint
# Check status with: scontrol show job <ID>
# Start from $SCRATCH
# See also: http://user.cscs.ch/get_started/run_batch_jobs/piz_daint/index.html
# SLURM_CPUS_PER_TASK= --cpus-per-task
# SLURM_NTASKS=--ntasks
##PBS -N GROMACS-multi
##PBS -l nodes=2:ppn=20
##PBS -q ivb20_k40
##PBS -l walltime=00:10:00
##PBS -m bea
##PBS -j oe
##PBS -M jkraus@nvidia.com
#======START===============================
START_TIME=`date`

#Setup environment
if [ "$1" == "daint" ]; then
	module load fftw
	module load cudatoolkit
	module switch PrgEnv-cray PrgEnv-gnu
	module load gcc
	source ${HOME}/local/gromacs-5.0-rc1-20140507_1204/bin/GMXRC.bash
elif [ "$1" == "PSG" ]; then
	source /home-2/jkraus/local/gromacs-5.0-rc1-20140507_0432/bin/GMXRC.bash
fi

MDRUN=`which mdrun_mpi`

MAXH=0.2
#MAXH=0.01

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
		echo "ERROR: PBS_NP%OMP_NUM_THREADS != 0 "
		exit 1
	fi 
	NP=$(( ${PBS_NP}/${OMP_NUM_THREADS} ))
	if [ $(( ${NP}%${PBS_NUM_NODES} )) -ne 0 ]; then
		echo "ERROR: NP%PBS_NUM_NODES != 0"
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
				echo "ERROR: NPME%PBS_NUM_NODES != 0"
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
		LOG_FILE_NAME="${input_basename}.${NB}.${NPME}.${PBS_NUM_NODES}.${NP}x${OMP_NUM_THREADS}.md.log"
	fi 
	ENER_FILE_NAME="${LOG_FILE_NAME}.ener"
	
	cmd="mpirun --display-map --report-bindings ${MAPPING_POLICY} -np ${NP} ${MDRUN} -ntomp ${OMP_NUM_THREADS} -nstlist ${NSTLIST} -npme ${NPME} ${GPU_ID_CMD} -nb ${NB} -s ${INPUT} -g ${LOG_FILE_NAME} -e ${ENER_FILE_NAME} -maxh ${MAXH} -resethway -noconfout -v"
	echo "${cmd}"
	${cmd}
}

#Relies on SLURM_CPUS_PER_TASK, SLURM_NTASKS and SLURM_JOB_NUM_NODES 
#ARGS
# $1 NPME				Number of separate ranks to be used for PME, -1 is guess
# $2 NSTLIST			Set nstlist when using a Verlet buffer tolerance (0 is guess)
# $3 NB					Calculate non-bonded interactions on: auto, cpu, gpu, gpu_cpu
# $4 INPUT				Input tpr file
# $5 LOG_FILE_NAME		name for the GROMACS log file
function launch_gromacs_slurm_cray_xc30 {
	export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}	#Number of OpenMP threads per MPI rank
	if [ $(( ${SLURM_NTASKS}%${SLURM_JOB_NUM_NODES} )) -ne 0 ]; then
		echo "ERROR: SLURM_NTASKS%SLURM_JOB_NUM_NODES != 0"
		exit 1
	fi 
	PPN=$(( ${SLURM_NTASKS}/${SLURM_JOB_NUM_NODES} ))

	GPU_ID_CMD=""
	NPPPN=${PPN}
	NPME=$1										#Number of separate ranks to be used for PME, -1 is guess
	NSTLIST=$2 	#Set nstlist when using a Verlet buffer tolerance (0 is guess)
	NB=$3		#Calculate non-bonded interactions on: auto, cpu, gpu, gpu_cpu 
	if [ "${NB}" != "cpu" ] ; then
		if [ ${NPME} -gt 0 ]; then
			if [ $(( ${NPME}%${SLURM_JOB_NUM_NODES} )) -ne 0 ]; then
				echo "ERROR: NPME%SLURM_JOB_NUM_NODES != 0"
				exit 1
			fi
			NPMEPN=$(( ${NPME}/${SLURM_JOB_NUM_NODES} ))	# Number of PME ranks per node
			NPPPN=$(( ${PPN} - ${NPMEPN} ))					# Number of PP ranks per node
		fi
		
		if [ ${NPPPN} -gt 1 ]; then
			echo "GPUs are shared between processes activating MPS"
			export CRAY_CUDA_PROXY=1 
			if [ ${PPN} -le 8 ]; then
				#CUDA_DEVICE_MAX_CONNECTIONS=16 leads to a seg fault!
				export CUDA_DEVICE_MAX_CONNECTIONS=${PPN}
			fi
			GPU_ID_CMD=" -gpu_id `for (( pp=1; pp<=${NPPPN}; pp++ )); do printf 0; done` "
		fi
	fi
	
	INPUT=$4
	
	if [ $5 ]; then
		LOG_FILE_NAME=$5
	else
		input_basename=$(basename "${INPUT}" .tpr)
		LOG_FILE_NAME="${input_basename}.${NB}.${NPME}.${SLURM_JOB_NUM_NODES}.${SLURM_NTASKS}x${OMP_NUM_THREADS}.md.log"
	fi 
	ENER_FILE_NAME="${LOG_FILE_NAME}.ener"
	
	cmd="aprun -cc none -n ${SLURM_NTASKS} -d ${OMP_NUM_THREADS} -N ${PPN} ${MDRUN} -ntomp ${OMP_NUM_THREADS} -pin on -nstlist ${NSTLIST} -npme ${NPME} ${GPU_ID_CMD} -nb ${NB} -s ${INPUT} -g ${LOG_FILE_NAME} -e ${ENER_FILE_NAME} -maxh ${MAXH} -resethway -noconfout -v"
	echo "${cmd}"
	${cmd}
	echo "Slurm Job ID: ${SLURM_JOBID} was executed on " >> ${LOG_FILE_NAME}
	echo $SLURM_JOB_NODELIST >> ${LOG_FILE_NAME}
}

if [ "$1" == "daint" ]; then
	cd ${SCRATCH}/GROMACS/run/KTH/lignocellulose
	INPUT=${SCRATCH}/GROMACS/run/KTH/data/lignocellulose.tpr
	if [ ! -f ${INPUT} ]; then 
		echo "TODO: add grompp call to generate topol.tpr"
		#aprun -n1 `which grompp_mpi` -f pme.mdp
	fi
elif [ "$1" == "PSG" ]; then
	#Set Boost Clocks for GPUs (Values assume max boost for Tesla K40)
	mpirun --map-by ppr:1:node -np $PBS_NUM_NODES nvidia-smi -ac 3004,875
	INPUT=/home-2/jkraus/workspace/GROMACS/run/KTH-Cray-RFP/data/lignocellulose.tpr
fi

if [ "$1" == "daint" ]; then
	PPN=$(( ${SLURM_NTASKS}/${SLURM_JOB_NUM_NODES} ))
	#                              NPME NSTLIST NB  INPUT    LOG_FILE_NAME
	#launch_gromacs_slurm_cray_xc30 0    0       gpu ${INPUT}
	##                              NPME NSTLIST NB  INPUT    LOG_FILE_NAME
	#launch_gromacs_slurm_cray_xc30 -1   0       cpu ${INPUT}
	#cp *.md.log /users/jkraus/workspace/GROMACS/benchmarks/KTH
	#if [ ${PPN} -ge 4 ]; then
	#	for (( npmepn=$(( ${PPN}/4 )); npmepn<=$(( ${PPN}/3 )); npmepn++ )); do
	#		npme=$(( ${npmepn}*${SLURM_JOB_NUM_NODES} ))
	#		#                              NPME    NSTLIST NB  INPUT    LOG_FILE_NAME
	#		launch_gromacs_slurm_cray_xc30 ${npme} 0       gpu ${INPUT}
	#	done
	#fi
	#                              NPME NSTLIST NB  INPUT    LOG_FILE_NAME
	#launch_gromacs_slurm_cray_xc30 -1   0       cpu ${INPUT}
	#                              NPME NSTLIST NB  INPUT    LOG_FILE_NAME
	if [ ${PPN} -eq 8 ]; then
		npmepn=3 #$(( ${PPN}/4 ))
		npme=$(( ${npmepn}*${SLURM_JOB_NUM_NODES} ))
		#                              NPME    NSTLIST NB  INPUT    LOG_FILE_NAME
		launch_gromacs_slurm_cray_xc30 ${npme} 0       gpu ${INPUT}
	fi
	cp *.md.log /users/jkraus/workspace/GROMACS/benchmarks/KTH/lignocellulose
elif [ "$1" == "PSG" ]; then
	for ompt in 1 5 10 20; do
		#                           OMP_NUM_THREADS NPME NSTLIST NB  INPUT    LOG_FILE_NAME
		launch_gromacs_pbs_openmpi  ${ompt}         0    0       gpu ${INPUT}
		#launch_gromacs_pbs_openmpi ${ompt}         -1   0       cpu ${INPUT}
	done
fi

if [ "$1" == "daint" ]; then
	cd ${SCRATCH}/GROMACS/run/KTH/ion_channel
	INPUT=${SCRATCH}/GROMACS/run/KTH/data/ion_channel.tpr
	if [ ! -f ${INPUT} ]; then 
		echo "TODO: add grompp call to generate topol.tpr"
		#aprun -n1 `which grompp_mpi` -f pme.mdp
	fi
elif [ "$1" == "PSG" ]; then
	INPUT=/home-2/jkraus/workspace/GROMACS/run/KTH-Cray-RFP/data/ion_channel.tpr
fi

echo "Runtime ${START_TIME} - `date`"
#======END=================================