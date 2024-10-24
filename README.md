# rTsfNet

It is a reference implementation of rTsfNet: multi-head rotation and time series feature net.

# LICENSE
Use of rTsfNet in publications must be acknowledged by referencing the following publication. 

- Yu Enokibori. 2023. rTsfNet: a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition. arXiv:2310.19283 [cs.HC]
- https://doi.org/10.48550/arXiv.2310.19283

## NOTE
The DOI and citation text will be updated for formal one. So, please check here before you make a submission.

## Paper info. 
### Title
rTsfNet: a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition
### Authors
Yu Enokibori, Nagoya University, Japan
### Abstruct
This paper proposes rTsfNet, a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction, as a new DNN model for IMU-based human activity recognition (HAR). rTsfNet automatically selects 3D bases from which features should be derived by deriving 3D rotation parameters within the DNN. Then, time series features (TSFs), the wisdom of many researchers, are derived and realize HAR using MLP. Although a model that does not use CNN, it achieved the highest accuracy than existing models under well-managed benchmark conditions and multiple datasets: UCI HAR, PAMAP2, Daphnet, and OPPORTUNITY, which target different activities.


# Best Results and Models
The best results are summarized in IMU-based HAR Benchmark ( https://bit.ly/45OZ1aT )
The trained models for the best results are available at
- https://bit.ly/46GWu3L

With the benchmark system, you can run rTsfNet like:
```
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'tsf' --boot_strap_epochs 150 --patience 50 --epochs 350
```

- More than 150 or so epochs of bootstrap protection should be used.

# Benchmark setup
This program can be worked with the following benchmark system.
Simply marge this repository for the benchmark system, then select 'tsf' for the model name.

IMU-based HAR Benchmark
- https://bit.ly/45OZ1aT

At least, this source code can work with the above benchmark system and the following packages.
```
nvidia-cuda-runtime-cu12  12.2.128                 pypi_0    pypi
nvidia-cublas-cu12        12.2.4.5                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
optuna                    3.2.0                    pypi_0    pypi
pandas                    2.0.3                    pypi_0    pypi
python                    3.9.16          h2782a2a_0_cpython    conda-forge
scikit-learn              1.3.0                    pypi_0    pypi
tensorboard               2.13.0                   pypi_0    pypi
tensorflow                2.13.0                   pypi_0    pypi
```

# Suggestion: How to optimize parameters

The authors recommend the following steps to optimize TSF set and paramters.
1. Optimize time series feature (TSF) set with temporal network structures.
2. Optimize parameters of network structure with the selected TSFs.
You can use --optuna option of the IMU-based HAR Benchmark.
like:
```
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'tsf' --boot_strap_epochs 150 --patience 50 --epochs 350 --optuna --optuna_study_suffix 20231030 --optuna_num_of_trial 600
```

- More than 300 or 600 of trials are suggested.
- More than 150 or so epochs of bootstrap protection should be used.

# Note
Tensorflow 2.15 has [a bug](https://github.com/tensorflow/tensorflow/issues/62607) on LayerNormalization.
So please use other versions or set 1e-7 for the epsilon attribute of LayerNormalization.
