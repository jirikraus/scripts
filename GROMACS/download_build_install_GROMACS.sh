#!/bin/bash

#Set Paths and URLS
DATE_STRING="`date +%Y%m%d_%H%M`"
#GROMACS_VERSION="5.0-rc1"
GROMACS_VERSION="4.6.5"
LOG_FILE="`pwd`/GROMACS-${GROMACS_VERSION}-install-${DATE_STRING}.log"
GROMACS_ARCHIVE=gromacs-${GROMACS_VERSION}.tar.gz
GROMACS_DOWNLOAD_URL=ftp://ftp.gromacs.org/pub/gromacs/gromacs-${GROMACS_VERSION}.tar.gz
GROMACS_SRC_DIR=`pwd`/gromacs-${GROMACS_VERSION}
GROMACS_BUILD_DIR=`pwd`/gromacs-${GROMACS_VERSION}-${DATE_STRING}
GROMACS_BUILD_DIR_MPI=`pwd`/gromacs-${GROMACS_VERSION}-${DATE_STRING}-mpi
GROMACS_INSTALL_DIR=${HOME}/local/gromacs-${GROMACS_VERSION}-${DATE_STRING}
BUILD_MPI_VERSION=1
BUILD_NONE_MPI_VERSION=1

#Setup Environment
module load cmake
module load cuda/6.0.37/toolkit
module load gcc/4.8.2
module load mpi/gnu/4.8.2/openmpi/1.8-cuda6.0.37

## Should be no need to modify beyond this point

#Report environment
module list 2>&1 | tee ${LOG_FILE}
set 2>&1 | tee -a ${LOG_FILE}
cat /proc/cpuinfo | sort | uniq 2>&1 | tee -a ${LOG_FILE}

#Download, Configure, Build and Install
#wget ${GROMACS_DOWNLOAD_URL} 2>&1 | tee -a ${LOG_FILE}
#tar -xzvf ${GROMACS_ARCHIVE} 2>&1 | tee -a ${LOG_FILE}

if [ ${BUILD_NONE_MPI_VERSION} -ne 0 ] ; then

	echo "Building and installing MPI Version of GROMACS" 2>&1 | tee -a ${LOG_FILE}
	mkdir -p ${GROMACS_BUILD_DIR}
	cd ${GROMACS_BUILD_DIR}

	echo "CC=gcc CXX=g++ cmake ${GROMACS_SRC_DIR} -DGMX_OPENMP=ON -DGMX_GPU=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_PREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GROMACS_INSTALL_DIR}" 2>&1 | tee -a ${LOG_FILE}
	CC=gcc CXX=g++ cmake ${GROMACS_SRC_DIR} -DGMX_OPENMP=ON -DGMX_GPU=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_PREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GROMACS_INSTALL_DIR} 2>&1 | tee -a ${LOG_FILE}
	echo "make install" 2>&1 | tee -a ${LOG_FILE}
	make install 2>&1 | tee -a ${LOG_FILE}

	cd ..

fi

if [ ${BUILD_MPI_VERSION} -ne 0 ] ; then
	echo "Building and installing MPI Version of GROMACS" 2>&1 | tee -a ${LOG_FILE}
	mkdir -p ${GROMACS_BUILD_DIR_MPI}
	cd ${GROMACS_BUILD_DIR_MPI}
	echo "CC=mpicc CXX=mpiCC cmake ${GROMACS_SRC_DIR} -DGMX_BUILD_MANPAGES=OFF -DGMX_OPENMP=ON -DGMX_GPU=ON -DGMX_MPI=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_PREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GROMACS_INSTALL_DIR}" 2>&1 | tee -a ${LOG_FILE}
	CC=mpicc CXX=mpiCC cmake ${GROMACS_SRC_DIR} -DGMX_BUILD_MANPAGES=OFF -DGMX_OPENMP=ON -DGMX_GPU=ON -DGMX_MPI=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_PREFER_STATIC_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GROMACS_INSTALL_DIR} 2>&1 | tee -a ${LOG_FILE}
	echo "make install" 2>&1 | tee -a ${LOG_FILE}
	make install 2>&1 | tee -a ${LOG_FILE}
	cd ..
fi

echo "Done" | tee -a ${LOG_FILE}
