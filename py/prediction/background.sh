# Delete existing screen
for session in $(sudo screen -ls | grep -o '[0-9]*.smell-pittsburgh-prediction')
do
  sudo screen -S "${session}" -X quit
  sleep 2
done

# Delete the log
sudo rm screenlog.0

# For python in conda env
#sudo screen -dmSL "smell-pittsburgh-prediction" bash -c ". '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate smell-pittsburgh-prediction; python main.py pipeline"
sudo screen -dmSL "smell-pittsburgh-prediction" bash -c ". '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate smell-pittsburgh-prediction; python main.py analyze"

# For globally installed python
#sudo screen -dmSL "smell-pittsburgh-prediction" python main.py pipeline
#sudo screen -dmSL "smell-pittsburgh-prediction" python main.py analyze

sudo screen -ls
