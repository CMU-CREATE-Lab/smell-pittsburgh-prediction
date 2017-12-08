sudo screen -X quit
sudo rm screenlog.0
sudo screen -dmSL "smell-pittsburgh-analysis" python main.py "run_all"
sudo screen -ls
