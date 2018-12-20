# smell-pittsburgh-prediction
A tool for predicting and interpreting smell data obtained from Smell Pittsburgh. The design and evaluation are documented in the paper, [Community-Empowered Mobile Smell Reporting System](https://arxiv.org/abs/1810.11143). If you find this useful, please consider citing:<br/>

Yen-Chia Hsu, Jennifer Cross, Paul Dille, Michael Tasota, Beatrice Dias, Randy Sargent, Ting-Hao (Kenneth) Huang, and Illah Nourbakhsh. 2018. Community-Empowered Mobile Smell Reporting System. arXivpreprint arXiv:1810.11143. 
```
@article{hsu2018community,
  title={Community-Empowered Mobile Smell Reporting System},
  author={Hsu, Yen-Chia and Cross, Jennifer and Dille, Paul and Tasota, Michael and Dias, Beatrice and Sargent, Randy and Huang, Ting-Hao'Kenneth' and Nourbakhsh, Illah},
  journal={arXiv preprint arXiv:1810.11143},
  year={2018}
}
```
# Usage
1. Install conda. A detailed documentation is [here](https://conda.io/docs/user-guide/getting-started.html). To use the following script, first visit [here](https://www.anaconda.com/download/#linux) to obtain the downloading path.
```sh
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh # replace this with the desired path
sudo sh Anaconda3-5.3.1-Linux-x86_64.sh
# hold the “d” key on keyboard to go through agreement
# when prompted to enter the installation location, use “/opt/anaconda3”

sudo vim /etc/profile
# add the following two lines to /etc/profile
export PATH="/opt/anaconda/bin:$PATH"
. /opt/anaconda3/etc/profile.d/conda.sh

source /etc/profile
conda update conda
conda deactivate
```
2. Clone this repository
```sh
git clone https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction.git
```
3. Create and activate conda environment
```sh
conda create -n smell-pittsburgh-prediction
source ~/.bashrc # only run this for Linux-based operation systems
conda activate smell-pittsburgh-prediction
```
4. Install packages in the conda environment. The install_packages script is default for Mac and Linux. For Windows, please open the script with a text editor, comment out the first line, and uncomment the last line.
```sh
cd smell-pittsburgh-prediction/
sh install_packages.sh
```
5. Get data, preprocess data, extract features, train the classifier, perform cross validation, analyze data, and interpret the model. This will create a directory (py/prediction/data_main/) to store all downloaded and processed data.
```sh
cd py/prediction/
python main.py pipeline

# For each step in the pipeline
python main.py data # get data
python main.py preprocess # preprocess data
python main.py feature # extract features
python main.py validation # perform cross validation
python main.py analyze # analyze data and interpret model
```
6. Train the classifier for production
```sh
cd py/prediction/
python production.py train
```
7. Make a prediction for production
```sh
cd py/prediction/
python production.py predict
```

# Visualization
The web/GeoHeatmap.html visualizes distribution of smell reports by zipcodes. You can open this by using a browser, such as Google Chrome.

# Dataset
A pre-downloaded dataset from 10/31/2016 to 9/30/2018 is included in this repository. To get recent data, change the end_dt (ending date time) variable in the main.py file and then run the following:
```sh
cd py/prediction/
python main.py data
```
This will download smell data (py/prediction/data_main/smell_raw.csv) and sensor data (py/prediction/data_main/esdr_raw/). The smell data is obtained from [Smell Pittsburgh](https://github.com/CMU-CREATE-Lab/smell-pittsburgh-rails/wiki/How-to-use-the-API). The sensor data is obtained from [ESDR](https://github.com/CMU-CREATE-Lab/esdr/blob/master/HOW_TO.md).
