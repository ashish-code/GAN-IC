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

## Project development overview

We worked on Homographynet work and the Spatial Transformer network towards achieving our goal of effectively splicing the donor image ROI into the recipient image in previous reports. Here we discuss our development of the network architecture, the training regime, and hyper-parameter optimization.

With a focus on visual category of people in images we explore the ability of our spatial transformer network to compute an effective homography for ‘glasses’ onto ‘human faces’. We have worked with face images in the past so the CelebA dataset makes sense as a training set. We focus on the cropped and aligned subset of faces from CelebA. This constraint of pose obviously will be a problem with unconstrained face images in the wild during testing, however, it will allow us to build
the network to handle (a) Scale; (b) Rotation; (c) and Translation parameters. We will augment the network with pose variation to extend the network to train on (d) Shear and (e) Projective parameters. Results use the following sample of human face and glasses. The glasses image is manually segmented. With development of image segmentation the donor
image will be algorithmicaly acquired from donor images in the wild.

<img src="https://github.com/ashish-code/GAN-IC/blob/master/media/image1.png" width="150" height="200">

<img src="https://github.com/ashish-code/GAN-IC/blob/master/media/image2.png" width="150" height="200">

Some preliminary results from our work on splicing 'glasses' onto 'faces':

The first image is 'input', both face and the glasses are displayed with the glasses at their 'initial' position, scale, and rotation. The second image is the 'output', the glasses have been warped onto their 'correct' position onto the face.

<img src="https://github.com/ashish-code/GAN-IC/blob/master/media/image3.png" width="150" height="150">

I've used my face, which is not part of the celebA dataset. It is also not equivalent to the cropped+aligned+centered faces that are used in the training of the GAN. Actually, this divergence in the test sample (my face) from the entire corpus of training images is reflected in some results where the glasses are bizarrely warped.

The failure of such a scenario:

<img src="https://github.com/ashish-code/GAN-IC/blob/master/media/image5.png" width="150" height="150">

In this failure case, the discriminator (part of GAN) seems to show is limitations. Since, its trained for celebA cropped and aligned faces, it can happen that a 'variation' in the input data falls in the 'dark zone' of the parameter space. In other words, the discriminator has never seen sufficient training data for this space and essentially makes a 'guess',
an extrapolation. Such extrapolation can lead to divergence in the feedback to the affine transform parameters (scale, rotation, translation). Hence, the glasses are erroneously warped.

#### More results
<div class="row">
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/image_g0_output.png" width="150">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/image_g2_output.png" width="150">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/image_g6_output.png" width="150">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/image_g7_output.png" width="150">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/image_g9_output.png" width="150">
  </div>
</div>


Faces in the wild
-----------------

We extended our approach to function with faces in the wild, i.e. unconstrained faces in terms of number of faces in the recipient image and their pose. Naturally this requires detection of face(s) in the recipient image. We incorporated Multi-Task Cascaded Convolutional Networks (MTCNN) towards this task. Our aim was to extract a bounding-box of the faces such that the face is correlated to the cropped and aligned faces of CelebA dataset used in training the spatial transform GAN. Obviously, the faces in the will not be aligned and this will be an issue we will progressively resolve.

### Face detection

Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations and occlusions. Recent studies show that deep learning approaches can achieve impressive performance on these two tasks.

<img src="https://github.com/ashish-code/GAN-IC/blob/master/media/image7.png" width="1000" height="150">

The MTCNN is a deep cascaded multi-task framework which exploits the inherent correlation between detection and alignment to boost up their performance. In particular, it leverages a cascaded architecture with three stages of carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. The results of the network depend on consistent facial landmarks and the consequent bounding box is a subset of the actual face. For the moment we simply extend the bounding box proportionally to human facial dimensions, assuming that detected faces are reasonably oriented, i.e.
the person in generally upright in the image.

### Glasses on Faces

In our sample images we typically have multiple people, each with different orientations. We detect the face and utilize the cropped image with our spatial transformer GAN to splice the glasses onto the face. The resulting face with glasses is reinserted into the test recipient image. Some results using multiple faces and multiple glasses are shown below:

<div class="row">
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/scienceguys.png" alt="Science" width="400">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/on_science.png" alt="on science" width="400">
  </div> 
</div>

<div class="row">
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/oscars.png" alt="Science" width="400">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/on_oscars.png" alt="on science" width="400">
  </div> 
</div>

<div class="row">
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/solvay.png" alt="Science" width="500">
  </div>
  <div class="column">
    <img src="https://github.com/ashish-code/GAN-IC/blob/master/images/on_solvay.png" alt="on science" width="500">
  </div> 
</div>

