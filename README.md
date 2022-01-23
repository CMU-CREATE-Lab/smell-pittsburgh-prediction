# smell-pittsburgh-prediction
A tool for predicting and interpreting smell data obtained from [Smell Pittsburgh](https://smellpgh.org/). The code for the system can be found [here](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-rails). The design, evaluation, and data analysis are documented in the following paper. If you find this useful, we would greatly appreciate it if you could cite the paper below.

Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2020. Smell Pittsburgh: Engaging Community Citizen Science for Air Quality. ACM Transactions on Interactive Intelligent Systems. 10, 4, Article 32. DOI:[https://doi.org/10.1145/3369397](https://doi.org/10.1145/3369397). Preprint:[https://arxiv.org/pdf/1912.11936.pdf](https://arxiv.org/pdf/1912.11936.pdf).

# Usage
Install conda. This assumes that Ubuntu is installed. A detailed documentation is [here](https://conda.io/docs/user-guide/getting-started.html). First visit [here](https://conda.io/miniconda.html) to obtain the downloading path. The following script install conda for all users:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh
sudo sh Miniconda3-py38_4.8.2-Linux-x86_64.sh -b -p /opt/miniconda3

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

# Run the entire pipeline
python main.py pipeline

# For each step in the pipeline
python main.py data # get data
python main.py preprocess # preprocess data
python main.py feature # extract features
python main.py validation # perform cross validation
python main.py analyze # analyze data and interpret model
```
To deploy the model and generate push notifications when smell events are predicted, run the following:
```sh
# Train the classifier
python production.py train

# Perform prediction
python production.py predict
```
If you want to disable the crowd-based smell event notifications, go to the "production.py" file and comment out the following line:
```python
if y_pred in (2, 3): pushType2(end_dt, logger)
```
Use [crontab]((https://help.ubuntu.com/community/CronHowto)) to call the above two commands periodically. The following example re-trains the model on every Sunday at 0:00. The prediction task is performed between 5:00 and 13:00 for each day at the 0 and 30 minutes clock (e.g., 5:00, and 5:30).
```sh
sudo crontab -e

# Add the following lines in the crontab file
0 0 * * 0 export PATH="/opt/miniconda3/bin:$PATH"; . "/opt/miniconda3/etc/profile.d/conda.sh"; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py train
0 5-13 * * * export PATH="/opt/miniconda3/bin:$PATH"; . "/opt/miniconda3/etc/profile.d/conda.sh"; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py predict
15 5-13 * * * export PATH="/opt/miniconda3/bin:$PATH"; . "/opt/miniconda3/etc/profile.d/conda.sh"; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py predict
30 5-13 * * * export PATH="/opt/miniconda3/bin:$PATH"; . "/opt/miniconda3/etc/profile.d/conda.sh"; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py predict
45 5-13 * * * export PATH="/opt/miniconda3/bin:$PATH"; . "/opt/miniconda3/etc/profile.d/conda.sh"; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py predict
```
IMPORTANT: the above crontab commands only work in bash, not shell. Make sure that you add the following at the first line in the crontab:
```sh
SHELL=/bin/bash
```
We can simplify the crontab as shown below. This means that we are running the command every 15 minutes. Check [this website](https://crontab.guru/every-15-minutes) for setting up the crontab.
```sh
sudo crontab -e

# Add the following lines in the crontab file
0 0 * * 0 export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py train
*/15 5-13 * * * export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate smell-pittsburgh-prediction; cd /var/www/smell-pittsburgh-prediction/py/prediction; run-one python production.py predict
```

# Visualization
The web/GeoHeatmap.html visualizes distribution of smell reports by zipcodes. You can open this by using a browser, such as Google Chrome.

# Dataset
There are two datasets in this repository. [Version one](/dataset/v1) is the dataset that we used in the paper. [Version two](/dataset/v2) is an updated dataset that covers a wider range of geographical regions and time range.

To get recent data, change the end_dt (ending date time) variable in the main.py file and then run the following:
```sh
python main.py data
```
This will download smell data (py/prediction/data_main/smell_raw.csv) and sensor data (py/prediction/data_main/esdr_raw/). The smell data is obtained from [Smell Pittsburgh](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-rails/wiki/How-to-use-the-API). The sensor data is obtained from [ESDR](https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md).
