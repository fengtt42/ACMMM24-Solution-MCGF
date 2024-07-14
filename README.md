# Multi-weather Cross-view Geo-localization Using Denoising Diffusion Models

This is the code repository of UAVM Workshop solutions of ACM MM24.

## Multi-weather Cross-view Geo-localization
Cross-view geo-localization depends on finding the correct location by matching drone-view images with satellite-view images. Weather variants, including fog, rain, snow, and multiple weather compositions, are randomly sampled to increase the difficulty of geo-localization. The red box represents the correct match we want to achieve regardless of weather conditions.
![image](https://github.com/fengtt42/ACMMM24-Solution-MCGF/blob/master/imgs/1.png)

## Models
MCGF establishes a joint optimization between image restoration and geo-localization using denoising diffusion models. For image restoration, MCGF incorporates a shared encoder and a lightweight restoration module to help the backbone eliminate weather-specific information. For geo-localization, MCGF uses EVA-02 as a backbone for feature extraction, with cross-entropy loss for training and cosine distance for testing.
![image](https://github.com/fengtt42/ACMMM24-Solution-MCGF/blob/master/imgs/2.png)

## Prerequisites
- Python 3.6
- GPU Memory >= 8G
- Numpy > 1.12.1
- Pytorch 0.3+
- scipy == 1.2.1

## Getting started
### Dataset & Preparation
Download [University-160k-WX](https://hdueducn-my.sharepoint.com/personal/wongtyu_hdu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fwongtyu%5Fhdu%5Fedu%5Fcn%2FDocuments%2FDatasets%2Funiversity%5F160k%5Fwx%5Ftest%5Fset&ga=1) upon request. The dataset format is shown in the Datasets folder.

## Train & Evaluation
### Train & Evaluation University-160k-WX
```  
sh run_train.sh
sh run_test.sh
sh run_predict.sh
```
:sparkles:[Download The Trained Model](https://cloud.tsinghua.edu.cn/library/4ae612a5-4f38-4ad0-970b-b7c8cf0c720f/0%20-%20%E4%B8%B4%E6%97%B6%E6%96%87%E4%BB%B6/net_009.pth)

## Results_1
> All test results can be displayed and downloaded on the competition result submission platform.
|Methods|R@1|R@5|R@10|AP|
|---|---|---|---|---|
|LPN|7.98|10.25|11.21|8.49|
|MBEG|26.17|32.84|35.32|29.32|
|Muse-Net|50.48|63.19|67.34|53.27|
|MCGF(ours)|84.68|91.36|93.00|88.71|


## results_2
![image](https://github.com/fengtt42/ACMMM24-Solution-MCGF/blob/master/imgs/3.png)


## Citation


