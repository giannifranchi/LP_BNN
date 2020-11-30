<p align="center"><img width="40%" src="./imgs/pytorch.png"></p>

LP-BNN CIFAR-10, CIFAR-100 official implementation using PyTorch
BatchEnsemble CIFAR-10, CIFAR-100 **unofficial** implementation using PyTorch

Please if you use this code please cite the following papers:
- LP-BNN
- BatchEnsemble
- [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v2.pdf)



## Requirements
see the requirement of [CIFAR code](https://github.com/meliketoy/wide-resnet.pytorch)
In addition to this requirement our code needs a big GPU to have a big batch.
Our code where tested and implemented on a V100 tesla GPU thanks to Jeanzay cluster.
Please proceed to the installation of Cuda, and Pytorch to as explained on 
[PyTorch web page](https://pytorch.org/get-started/locally/) 
to be able to use our code.


## How to TRAIN the Deep Neural Network with LP-BNN
After you have cloned the repository, you can train each dataset of either cifar10, cifar100 by running the script below.
To have better results we advise you to perform several trainings(minimum 3).
```bash
 python main_LPBNN.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T0
 python main_LPBNN.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T1
 python main_LPBNN.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T2
```

## How to train the Deep Neural Network with LP-BNN BatchEnsemble
After you have cloned the repository, you can train each dataset of either cifar10, cifar100 by running the script below.
To have better results we advise you to perform several trainings(minimum 3).
```bash
 python main_BatchEnsemble.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T0
 python main_BatchEnsemble.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T1
 python main_BatchEnsemble.py --dataset [cifar10/cifar100] --dirsave_out BE_C10_T2
```

## How to evaluate the code
here are the comand line to test for CIFAR10.
For CIFAR100 please adapt the code 
```bash
 python  evaluate_uncertainty.py --algo 'BE' --dataset cifar10 --dirsave_out './checkpoint/cifar10/BE_C10_T'
 python  evaluate_uncertainty.py --algo 'LPBNN' --dataset cifar10 --dirsave_out './checkpoint/cifar10/LBPNN_C10_T'
```

## Implementation Details


| **Hyper-parameter** | **CIFAR-10**| **CIFAR-100** |
|:-----------------:|:-----------------:|:-------:|
| Ensemble size  J   |      4 | 4|
|  initial learning rate    |      0.1 | 0.1|
| batch size    |      128 | 128|
| lr decay ratio  |     0.1 | 0.1|
|  lr decay epochs  |      80, 160, 200 | 80, 160, 200|
| cutout  |     True | True|
| SyncEnsemble BN  |     False | False|
|  Size of the latent space $  |     32 | 32|



## CIFAR-10 Results
 
![alt tag](imgs/cifar10_image.png)

Below is the result of the test set accuracy for **CIFAR-10 dataset** training.

**Accuracy is the average of 3 runs**

| network           | Accuracy (%)  | AUC | AUPR | FPR-95-TPR | ECE  (%)   | cA(%) |cE (%) |
|:-----------------:|:-------:|:----------:|:-----:|:-----:|:------------:|:-----------:|:-----------:|
| LP-BNN |    **96.48**    |  0.9540   | 0.9731 |   0.132   | 0.0167 |   47.44   | 0.2909    |
| BatchEnsemble |    95.02    |   **0.9691**  | **0.9836** |   **0.103**   | **0.0094** |    **69.51**    |  0.2324     |


If you are interrested about the corrupted accuraccy and corrupted expected calibration error please download the dataset from
[CIFAR-10-C](https://github.com/hendrycks/robustness)