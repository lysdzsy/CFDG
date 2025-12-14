# CFDG
Causal Feature-aware Dynamic Graph Neural Network for Open-Set Domain Generalization Diagnosis in Multi-Sensor Systems
![framework](https://github.com/user-attachments/assets/2446debc-b9b1-4278-bb57-54580b8cc217)
# Dependencies
pytorch 1.12.0

python 3.9.19

numpy 1.23.0

pandas 2.2.2

pytorch-lightning 1.8.6

relpath 3.0.5

seaborn 0.13.2

torch 2.4.0

torch-geometric 2.6.0

torchvision 0.19.0

tqdm 4.66.5

# How to Train
python train.py --mode train

python train.py --mode test -d [dataset]
# Reference
This paper has been accepted for publication in Advanced Engineering Informatics. The following papers were used as references for this code.

1. TodyNet: Temporal dynamic graph neural network for multivariate time series classification 
2. Out-of-Distribution Generalization on Graphs via Progressive Inference

