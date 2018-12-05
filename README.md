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
source ~/.bashrc
conda activate smell-pittsburgh-prediction
```
4. Install packages in the conda environment
```sh
cd smell-pittsburgh-prediction/
sudo chmod 777 install_packages.sh # only run this if cloning to a directory that belongs to root
./install_packages.sh
```
5. Install pytorch from https://pytorch.org/ (this is a testing feature)
5. Train the classifier and perform cross validation
```sh
cd py/prediction/
python main.py run_all
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
The web/ directory shows distribution of smell reports by zipcodes.
