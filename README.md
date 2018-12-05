# smell-pittsburgh-prediction
A tool for predicting and interpreting smell data obtained from Smell Pittsburgh.

# Usage
1. Install conda from https://conda.io/docs/user-guide/getting-started.html
2. Clone this repository
```sh
git clone https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction.git

# For cloning to a directory that belongs to root, use the following:
sudo git clone https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction.git
sudo chmod -R 777 smell-pittsburgh-prediction/
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
sudo chmod 777 install_packages.sh # only run this if cannot run the next command
./install_packages.sh
```
5. Get data, preprocess data, extract features, train the classifier, and perform cross validation
```sh
cd py/prediction/
python main.py pipeline

# For each step in the pipeline
python main.py data # get data
python main.py preprocess # preprocess data
python main.py feature # extract features
python main.py validation # perform cross validation
python main.py analyze # analyze data
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
The web/GeoHeatmap.html visualizes distribution of smell reports by zipcodes.
