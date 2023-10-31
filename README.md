# rTsfNet

It is a reference implementation of rTsfNet: multi-head rotation and time series feature net.

# LICENSE
Use of rTsfNet in publications must be acknowledged by referencing the following publication. 

TBC: plz wait for Nov. 1st
- https://doi.org/xxxxx

## NOTE
The DOI will be updated for formal one. So, please check here before you make a submission.

## Paper info. 
### Title
rTsfNet: a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition
### Authors
Yu Enokibori, Nagoya University, Japan
### Abstruct
This paper proposes rTsfNet, a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction, as a new DNN model for IMU-based human activity recognition (HAR). rTsfNet automatically selects 3D bases from which features should be derived by deriving 3D rotation parameters within the DNN. Then, time series features (TSFs), the wisdom of many researchers, are derived and realize HAR using MLP. Although a model that does not use CNN, it achieved the highest accuracy than existing models under well-managed benchmark conditions and multiple datasets: UCI HAR, PAMAP2, Daphnet, and OPPORTUNITY, which target different activities.


# Best Results and Models
The best results and trained models of each dataest are available at
- https://bit.ly/46GWu3L
.

# Benchmark setup
This program can be worked with the following benchmark system.
Simply marge this repository for the benchmark system, then select 'tsf' for the model name.

IMU-based HAR Benchmark
- https://bit.ly/45OZ1aT

# Suggestion: How to optimize parameters

The author recommend the following steps to optimize TSF set and paramters.
1. Optimize time series feature (TSF) set with temporal network structures.
2. Optimize parameters of network structure with the selected TSFs.
You can use --optuna option of the IMU-based HAR Benchmark.