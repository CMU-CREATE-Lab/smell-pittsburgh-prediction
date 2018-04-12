sudo screen -X quit
sudo rm screenlog.0
#sudo screen -dmSL "smell-pittsburgh-prediction" python main.py "run_all"
sudo screen -dmSL "smell-pittsburgh-prediction" python main.py "analyze"
sudo screen -ls
