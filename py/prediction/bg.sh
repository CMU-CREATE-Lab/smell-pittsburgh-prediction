#!/bin/sh

# This script runs a python script using screen command
# For example:
# sh bg.sh python train.py i3d-rgb

# Get file path
if [ "$1" != "" ] && [ "$2" != "" ]
then
  echo "Run: $1 $2 $3 $4 $5 $6"
else
  echo "Usage examples:\n\
  sh bg.sh python main.py validation\n\
  sh bg.sh python main.py pipeline\n\
  sh bg.sh python main.py analyze"
  exit 1
fi

# Delete existing screen
for session in $(sudo screen -ls | grep -o "[0-9]*.$1.$2.$3")
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env in Ubuntu
sudo screen -dmSL "$1.$2.$3" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate smell-pittsburgh-prediction; $1 $2 $3 $4 $5 $6"

# List screens
sudo screen -ls
exit 0
