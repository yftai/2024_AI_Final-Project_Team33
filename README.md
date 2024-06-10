# AI Final Project Team33 - Trying Different Deep Learning Techniques to Enhance Eye Disease Detection

## Task Overview 
APTOS 2019 Blindness Detection(https://www.kaggle.com/c/aptos2019-blindness-detection).  
Aim to train a deep learning model to detect blindness before it happened.  
1. Preprocessing dataset
2. 1st-level models: train 3 models (resnext) on the same dataset
3. 2nd-level models: add deep learning techniques - pseudo labeling & ensemble
4. Evaluation


## Prerequisite
|Package                                  | Version           |
|---------------------------------------- | ------------------|
|pip                                      | 23.3.2            |
|python                                   | 3.10.13           |
|opencv-python                            | 4.10.0.82         |
|joblib                                   | 1.4.2             |
|matplotlib                               | 3.7.5             |
|numpy                                    | 1.26.4            |
|pandas                                   | 2.2.1             |
|torch                                    | 2.1.2             |
|torchvision                              | 0.16.2            |
|Pillow                                   | 9.5.0             |
|scikit-image                             | 0.22.0            |
|scikit-learn                             | 1.2.2             |
|tqdm                                     | 4.66.4            |
|pretrainedmodels                         | 0.7.4             |
|scipy                                    | 1.11.4            |


## Usage

### Train 1st-level models
We train our first stage model on local with following command.
Dataset: APTOS 2019 train dataset + [2015 dataset](<https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/97860#581042>)
```
python train.py --arch se_resnext50_32x4d
python train.py --arch se_resnext101_32x4d --batch_size 24
python train.py --arch senet154 --batch_size 16
```  
  
### Train 2nd-level models with pseudo labels and ensemble
We train our second stage model on kaggle notebook.
https://www.kaggle.com/code/derektai/ai-final-project-team-33-stage-2


## Hyperparameters
epochs: 30  
batch_size: depends on models  
learning-rate: 1e-3  
Loss: MSE  
Optimizer: SGD  
LR scheduler: CosineAnnealingLR  


## Experiment Results
We get following result.  
- Public leader board: 0.822275  
- Private leader board: 0.930273
