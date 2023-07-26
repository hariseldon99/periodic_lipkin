#!/bin/bash
#----------------------------------------------------------
# Job name
#PBS -N ipr_full_3
# queue select
#PBS -q workq
# Name of stdout output file (default)
#PBS -o ipr_full_3.out 
# stdout output file
#PBS -l select=1:ncpus=48
#PBS -j oe
#PBS -l walltime=168:00:00 
#----------------------------------------------------------
IPYNB=run3.ipynb

# Put conda source command in $HOME/.bashrc
# Create and populate your  conda environment and add the name below
CONDAENV=hpc

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



#----------------------------------------------------------
#Suppress spurious infiniband-related errors
export MPI_MCA_mca_base_component_show_load_errors=0
export PMIX_MCA_mca_base_component_show_load_errors=0


# Change to submission directory
cd $PBS_O_WORKDIR

#Start time
start=`date +%s.%N`

conda run -n ${CONDAENV} nbterm --run ${IPYNB}
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
SSHBIN=/usr/bin/ssh
LOGIN_NODE=kuhpchn
source ${PBS_O_HOME}/.config/telegram/telegram.conf

URL="https://api.telegram.org/bot${TOKEN}/sendMessage"
# Generate the telegram message  text
TEXT="${bell} PBS Job ${PBS_JOBNAME} exiting @ ${HOSTNAME}:${PBS_O_WORKDIR}. Job ID: ${PBS_JOBID}"
CMD='curl -s --max-time 10 --retry 5 --retry-delay 2 --retry-max-time 10  -d '\""chat_id=${CHATID}&text=${TEXT}&disable_web_page_preview=true&parse_mode=markdown"\"" ${URL}"
$SSHBIN $LOGIN_NODE $CMD 
#----------------------------------------------------------
