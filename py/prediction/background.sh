sudo screen -X quit
sudo rm screenlog.0
sudo screen -dmSL "smell-pittsburgh-prediction" python analysis.py "run_all"
sudo screen -ls
