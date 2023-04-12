#!/bin/bash
#SBATCH --job-name=lmg_offres
#This sets the name of the job

#SBATCH --partition=GPU
#This sets the partition to the GPU partition. Important for GPU jobs

#SBATCH --gres=gpu:1
#This allocates 1 GPU as a Global Resource (gres). Important for GPU jobs

#SBATCH --ntasks=32
#This sets the number of processes to 32. Change if needed

#SBATCH --cpus-per-task=1
#This allocates the number of cpus per tasks. If the number of tasks is 4, and cpus per task is 1, then slurm will assign 4 X 1 = 4 cpus to the job

#SBATCH --time=3:00:00 
#This allocates the walltime to 2 day. The program will not run for longer.

#SBATCH --qos=elevated 
#This sets the quality of service to 'elevated'

#SBATCH --mail-type=FAIL,END                                                                         
#SBATCH --mail-user="telegram:5545394160"                                                  

#Start time
start=`date +%s.%N`

let nprocs=${SLURM_NTASKS}*${SLURM_CPUS_PER_TASK}
source /usr/local/condaenv/bin/activate
conda activate ~/.hpc
export MKL_RUN_THREADS=1
export OMP_NUM_THREADS=${MKL_RUN_THREADS}
nbterm --run prethermalization_multparams.ipynb
conda deactivate

#End time
end=`date +%s.%N`

RUNTIME=$( echo "$end - $start" | bc -l )
echo '---------------------------------------------'
echo "Runtime: "$RUNTIME" sec"
echo '---------------------------------------------'
