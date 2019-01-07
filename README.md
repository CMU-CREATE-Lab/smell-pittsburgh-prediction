# smell-pittsburgh-prediction
A tool for predicting and interpreting smell data obtained from [Smell Pittsburgh](https://smellpgh.org/). The design and evaluation are documented in the paper, [Smell Pittsburgh: Community-Empowered Mobile Smell Reporting System](https://arxiv.org/abs/1810.11143). If you find this useful, please consider citing:<br/>

Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2018. Smell Pittsburgh: Community-Empowered Mobile Smell Reporting System. arXivpreprint arXiv:1810.11143. 
```
@article{hsu2018smellpittsburgh,
  title={Smell Pittsburgh: Community-Empowered Mobile Smell Reporting System},
  author={Hsu, Yen-Chia and Cross, Jennifer and Dille, Paul and Tasota, Michael and Dias, Beatrice and Sargent, Randy and Huang, Ting-Hao'Kenneth' and Nourbakhsh, Illah},
  journal={arXiv preprint arXiv:1810.11143},
  year={2018}
}
```
# Usage
Install conda. This assumes that Ubuntu is installed. A detailed documentation is [here](https://conda.io/docs/user-guide/getting-started.html). First visit [here](https://conda.io/miniconda.html) to obtain the downloading path. The following script install conda for all users:
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh
sudo sh Miniconda3-4.5.11-Linux-x86_64.sh -b -p /opt/miniconda3

sudo vim /etc/bash.bashrc
# Add the following lines to this file
export PATH="/opt/miniconda3/bin:$PATH"
. /opt/miniconda3/etc/profile.d/conda.sh

source /etc/bash.bashrc
```
For Mac OS, I recommend installing conda by using [Homebrew](https://brew.sh/).
```sh
brew cask install miniconda
echo 'export PATH="/usr/local/miniconda3/bin:$PATH"' >> ~/.bash_profile
echo '. /usr/local/miniconda3/etc/profile.d/conda.sh' >> ~/.bash_profile
source ~/.bash_profile
```
Clone this repository and create conda environment.
```sh
git clone https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction.git
sudo chown -R $USER smell-pittsburgh-prediction
conda create -n smell-pittsburgh-prediction
```
Install packages.
```sh
conda activate smell-pittsburgh-prediction
conda install pip
conda install python=2.7
sh smell-pittsburgh-prediction/install_packages.sh
```
Get data, preprocess data, extract features, train the classifier, perform cross validation, analyze data, and interpret the model. This will create a directory (py/prediction/data_main/) to store all downloaded and processed data.
```sh
cd smell-pittsburgh-prediction/py/prediction/
python main.py pipeline

# For each step in the pipeline
python main.py data # get data
python main.py preprocess # preprocess data
python main.py feature # extract features
python main.py validation # perform cross validation
python main.py analyze # analyze data and interpret model

# For deployment, train the classifier and perform prediction
# Use crontab to call the following two commands periodically
# (https://help.ubuntu.com/community/CronHowto)
python production.py train
python production.py predict
```

# Visualization
The web/GeoHeatmap.html visualizes distribution of smell reports by zipcodes. You can open this by using a browser, such as Google Chrome.

# Dataset
A pre-downloaded dataset from 10/31/2016 to 9/30/2018 is included in this repository. To get recent data, change the end_dt (ending date time) variable in the main.py file and then run the following:
```sh
python main.py data
```
This will download smell data (py/prediction/data_main/smell_raw.csv) and sensor data (py/prediction/data_main/esdr_raw/). The smell data is obtained from [Smell Pittsburgh](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-rails/wiki/How-to-use-the-API). The sensor data is obtained from [ESDR](https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md).
