# smell-pittsburgh-prediction
A tool for predicting and interpreting smell data obtained from [Smell Pittsburgh](https://smellpgh.org/). The code for the system can be found [here](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-rails). The design and evaluation are documented in the paper, [Smell Pittsburgh: Community-Empowered Mobile Smell Reporting System](https://arxiv.org/pdf/1810.11143.pdf). If you find this useful, please consider citing:<br/>

Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao Huang, and Illah Nourbakhsh. 2019. Smell Pittsburgh: community-empowered mobile smell reporting system. In Proceedings of the 24th International Conference on Intelligent User Interfaces (IUI 2019). ACM.

```
@inproceedings{hsu-2019-smellpgh,
 author = {Hsu, Yen-Chia and Cross, Jennifer and Dille, Paul and Tasota, Michael and Dias, Beatrice and Sargent, Randy and Huang, Ting-Hao (Kenneth) and Nourbakhsh, Illah},
 title = {Smell Pittsburgh: Community-empowered Mobile Smell Reporting System},
 booktitle = {Proceedings of the 24th International Conference on Intelligent User Interfaces},
 series = {IUI '19},
 year = {2019},
 isbn = {978-1-4503-6272-6},
 location = {Marina del Ray, California},
 pages = {65--79},
 numpages = {15},
 url = {http://doi.acm.org/10.1145/3301275.3302293},
 doi = {10.1145/3301275.3302293},
 acmid = {3302293},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {air quality, citizen science, community empowerment, machine learning, smell, sustainable HCI},
} 
```
# Usage
Install conda. This assumes that Ubuntu is installed. A detailed documentation is [here](https://conda.io/docs/user-guide/getting-started.html). First visit [here](https://conda.io/miniconda.html) to obtain the downloading path. The following script install conda for all users:
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
sudo sh Miniconda3-4.7.12.1-Linux-x86_64.sh -b -p /opt/miniconda3

sudo vim /etc/bash.bashrc
# Add the following lines to this file
export PATH="/opt/miniconda3/bin:$PATH"
. /opt/miniconda3/etc/profile.d/conda.sh

source /etc/bash.bashrc
```
For Mac OS, I recommend installing conda by using [Homebrew](https://brew.sh/).
```sh
brew cask install miniconda
echo 'export PATH="/usr/local/Caskroom/miniconda/base/bin:$PATH"' >> ~/.bash_profile
echo '. /usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh' >> ~/.bash_profile
source ~/.bash_profile
```
Clone this repository.
```sh
git clone https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction.git
sudo chown -R $USER smell-pittsburgh-prediction
```
Create conda environment and install packages. It is important to install python 3.8 and pip first inside the newly created conda environment.
```sh
conda create -n smell-pittsburgh-prediction
conda activate smell-pittsburgh-prediction
conda install python=3.8
conda install pip
which pip # make sure this is the pip inside the smell-pittsburgh-prediction environment
sh smell-pittsburgh-prediction/install_packages.sh
```
If the environment already exists and you want to remove it before installing packages, use the following:
```sh
conda env remove -n smell-pittsburgh-prediction
```
Get data, preprocess data, extract features, train the classifier, perform cross validation, analyze data, and interpret the model. This will create a directory (py/prediction/data_main/) to store all downloaded and processed data. Notice that if you change the is_regr parameter in the "main.py" file, you will need to run "python main.py feature" again to create a new set of features.
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
