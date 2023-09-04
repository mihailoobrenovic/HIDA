# HIDA
Official code for the paper "Learning domain invariant representations of heterogeneous image data" published in Machine Learning journal. 

## Requirements:
* Tensorflow 1.15

## Data
Download data.zip file from the following [link](https://seafile.unistra.fr/f/b94d2def090d449fb9a8/?dl=1). The zip file contains RESISC45 and EuroSAT datasets.

Put the zip file inside of the "src" folder and unpack it.

## Training SS-HIDA
Here is an example of running a training of semi-supervised domain adaptation model SS-HIDA with RESISC45 as source and EuroSAT as target dataset, with 6.25% of labelled target data (25 labelled target images per class). From "src" folder, run the following terminal command:
```
python run.py --usecase ss_hida --source resisc --target eurosat --threshold 6_25 --exp_name ss_hida_6_25_1
```
Other possible values for `threshold` parameter are `2_5` and `1_25` for 2.5% (10 images per class) and 1.5% (5 images per class) of labelled target data respectively.

## Training U-HIDA
Here is an example of running a training of unsupervised domain adaptation model U-HIDA with EuroSAT as source and RESISC45 as target dataset. From "src" folder, run the following terminal command:
```
python run.py --usecase u_hida --source eurosat --target resisc --exp_name u_hida_1
```
