#!/bin/bash
#----------------------------------------------------------
# Job name
#PBS -N pretherm
# queue select
#PBS -q workq
# Name of stdout output file (default)
#PBS -o pretherm_20230329.out 
# stdout output file
#PBS -j oe
#PBS -l walltime=24:00:00 
#----------------------------------------------------------
# Enter the path to the python script that you wish to run on the GPU node
IPYNB=prethermalization_multparams.ipynb

# Put conda source command in $HOME/.bashrc
# Create and populate your  conda environment and add the name below
CONDAENV=hpc

# Enter the name of the GPU host
#GPUHOST=kuhpcgn1
GPUHOST=kuhpcgn2

# This is the actual command that the GPU node will run. Adjust as needed
# Beware the order of the quotes!
GPUNODE_NTHREADS=1
GPUNODE_CMD='export OMP_NUM_THREADS='"$GPUNODE_NTHREADS"'; export MKL_NUM_THREADS='"$GPUNODE_NTHREADS"';cd '"$PBS_O_WORKDIR"';conda run -n '"$CONDAENV"' nbterm --run '"$IPYNB"''

#----------------------------------------------------------
#Suppress spurious infiniband-related errors
export MPI_MCA_mca_base_component_show_load_errors=0
export PMIX_MCA_mca_base_component_show_load_errors=0


# Change to submission directory
cd $PBS_O_WORKDIR

echo "Starting Command on GPU Node ${GPUHOST}:"
echo '---------------------------------------------------------------------------------------------------------------------------------------'
echo $GPUNODE_CMD
echo '---------------------------------------------------------------------------------------------------------------------------------------'

#Start time
start=`date +%s.%N`

SSHBIN=/usr/bin/ssh
$SSHBIN $GPUHOST $GPUNODE_CMD
# Kill stray python processes
$SSHBIN $GPUHOST killall python
$SSHBIN $GPUHOST killall nbterm
#End time
end=`date +%s.%N`

RUNTIME=$( echo "$end - $start" | bc -l )
echo '---------------------------------------------'
echo "Runtime: "$RUNTIME" sec"
echo '---------------------------------------------'

#----------------------------------------------------------
# Communicate job status to a telegram bot
#----------------------------------------------------------
# Create a telegram bot and get TOKEN, CHATID from telegram botfather: 
# See https://www.cytron.io/tutorial/how-to-create-a-telegram-bot-get-the-api-key-and-chat-id
# Put them into two environment variables TOKEN and CHATID in a config file and source itlike below
#----------------------------------------------------------
SHBIN=/usr/bin/ssh
LOGIN_NODE=kuhpchn
source ${PBS_O_HOME}/.config/telegram/telegram.conf

URL="https://api.telegram.org/bot${TOKEN}/sendMessage"
# Generate the telegram message  text
TEXT="${bell} PBS Job ${PBS_JOBNAME} exiting @ ${HOSTNAME}:${PBS_O_WORKDIR}. Job ID: ${PBS_JOBID}"
CMD='curl -s --max-time 10 --retry 5 --retry-delay 2 --retry-max-time 10  -d '\""chat_id=${CHATID}&text=${TEXT}&disable_web_page_preview=true&parse_mode=markdown"\"" ${URL}"
$SSHBIN $LOGIN_NODE $CMD 
#----------------------------------------------------------
