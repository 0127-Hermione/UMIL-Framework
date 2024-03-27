# A Universal Multiple Instance Learning Framework for Whole Slide Image Analysis

## Pre-requisites:
python 3.7.3
torch 1.7.0
cuda 11.0
openslide-python 1.1.2


## Download  raw WSI datasets
# download classical MIL benchmark datasets
 $ python download.py --dataset=mil
# download TCGA-NSCLC dataset
   https://portal.gdc.cancer.gov/
   or $ python download.py --dataset=tcga
# download Camelyon16 dataset
   https://camelyon16.grand-challenge.org/Data/
   or  $ python download.py --dataset=c16
# download PANDA dataset
   https://www.kaggle.com/c/prostate-cancer-grade-assessment/data
#  download SICAPv2 dataset
   https://data.mendeley.com/datasets/9xxm58dvs3/1

## patch extraction
python patch_extraction.py -m 1 -b 20 -d wsi_path -v tiff -t 15
or 
python wsi_patch_gen.py and filltering

## feature extraction
ResNet:
python feature_extraction.py --dataset=path --weights=ImageNet --norm_layer=batch
Simclr:
'''
train Simclr embedder:
  $ cd simclr
  $ python run.py --dataset=[DATASET_NAME]
Set flag `--multiscale=1` and flag `--level=low`

python feature_extraction.py --dataset=path --weights=20x
you could use the trained embedder from dsmil [Camelyon16](https://drive.google.com/drive/folders/14pSKk2rnPJiJsGK2CQJXctP7fhRJZiyn?usp=sharing) and [TCGA](https://drive.google.com/drive/folders/1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE?usp=sharing)  
'''

## Training 
python train_classical_mil.py
python train_tcga.py --dataset=TCGA-NSCLC
python train_model.py --dataset=Camelyon16 --num_classes=1 


## Folder structures
Data is organized in two folders, `WSI` and `datasets`. `WSI` folder contains the images and `datasets` contains the computed features.
```
root
|-- WSI
|   |-- dataset_name
|   |   |-- class_1
|   |   |   |-- slide1.tif
|   |   |   |-- ...
|   |   |-- class_2
|   |   |   |-- slide2.tif
|   |   |   |-- ...
```
Once patch extraction is performed, `sinlge` folder will appear.
```
root
|-- WSI
|   |-- dataset_name
|   |   |-- single
|   |   |   |-- class_1
|   |   |   |   |-- slide_1
|   |   |   |   |   |-- patch_1.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |-- ...
```
Once feature computing is performed, `DATASET_NAME` folder will appear inside `datasets` folder.
```
root
|-- datasets
|   |-- dataset_name
|   |   |-- class_1
|   |   |   |-- slide_1.csv
|   |   |   |-- ...
|   |   |-- class_2
|   |   |   |-- slide_1.csv
|   |   |   |-- ...

```
For each bag, there is a .csv file where each row contains the feature of an instance. The .csv is named as "_bagID_.csv" 
For binary classifier, use `1` for positive bags and `0` for negative bags. Use `--num_classes=1` at training.  
For multi-class classifier (`N` positive classes and one optional negative class), use `0~(N-1)` for positive classes. If you have a negative class (not belonging to any one of the positive classes), use `N` for its label. Use `--num_classes=N` (`N` equals the number of **positive classes**) at training.





