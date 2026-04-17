#!/bin/bash
# Code for running MONARCHS on a HPC system. This one is designed for use on Hawk, the supercomputer run by
# the Advanced Research Computing team at Cardiff University. This can be used as a template for running on other
# systems using Slurm - you may need to amend some parameters (e.g. the partition, from "-p compute" to whatever is
# on your system) to make it actually run.
# This script assumes that you have a Singularity container downloaded - see ``get_singularity_container.sh`` for
# how to do so. It also assumes that your model setup file is called "model_setup.py", and is in the same
# directory as the MONARCHS setup file that you are running from. Your system may use Apptainer instead of Singularity;
# their functionality and syntax are essentially the same.

#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH -o MONARCHS.log.%J
#SBATCH -e MONARCHS.err.%J
#SBATCH -J MONARCHS
#SBATCH -p compute
#SBATCH --time=24:00:00

# Load in singularity shell

module purge
module load singularity

MYPATH=$SLURM_SUBMIT_DIR
NCORES=$SLURM_NTASKS
RUNDIR=$PWD
FILENAME='/path/to/model_setup.py'
USERNAME='test_username'

echo ' NODES USED = '$SLURM_NODELIST
echo ' SLURM_JOBID = '$SLURM_JOBID
echo ' CORES = '$NCORES
echo ' PARTITION = '$SLURM_JOB_PARTITION # useful for checking if a job was run on the "dev" node or not

echo Time is `date`
echo Directory is `pwd`

cd $MYPATH
echo Start time is `date`
singularity exec "/home/$USERNAME/monarchs_latest.sif python run_monarchs.py" -i "${RUNDIR}/${FILENAME}"
echo End time is `date`