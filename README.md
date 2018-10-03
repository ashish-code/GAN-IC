# GAN-IC
Image composition using GAN in conjunction with Homography estimation using HomographyNet and application of affine warp using Spatial Transformer Networks. Subsequently, appearance attributes of spatial transformed images will be modified using image intrinsic attribute estimation and applied using associated GAN/VAE. 

## Prerequisites
This code is developed with Python3. It requires TensorFlow r1.0+. 
Implemented in Ubuntu 18.04 system with CUDA 9.2, Python 3.6.6, CUDNN 7.3.1


The dependencies can install by running:
pip install --upgrade numpy scipy termcolor tensorflow-gpu

## Dataset
We focus on the celebA dataset for faces. It has instances of faces with and without glasses.
Link to the dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Ensure the following files from the CelebA dataset are are extracted and stored in path in local machine:

(1) Aligned & cropped images
(2) Attribute annotations
(3) Train/val/test partitions

After downloading CelebA, run $python preprocess_celebA.py to convert the data according to the provided train/test split to files in the ".npy" format. These files will be used to create training and testing batches.

## Under Construction
At present we are experimenting with the GAN training and parameter optimization. The dataset can get large to memory management can be an issue. The repository will be updated when a satisfactory trained GAN model is available.
