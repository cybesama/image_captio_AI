# Automated Image Captioning Using Deep Learning: A Comprehensive Analysis of Image Captioning Models

Pradeep_Singh

_-Delhi Technological University_


## Abstract

In this project, we will develop an image captioning bot capable of generating descriptive captions for images by employing machine learning and deep learning techniques. The process involves training a neural network model that can learn visual features from images and map them to relevant text sequences. The goal is to accurately predict meaningful captions for a wide range of images, capturing both objects and contextual information. We will use pre-existing large-scale datasets like Flicker 8K for training, with image feature extraction performed using a convolutional neural network (CNN), while the text generation will be handled by a recurrent neural network (RNN) or transformers. Additionally, we will explore Early and Late fusion techniques to combine visual and textual data and compare their performance in generating captions. This model will serve as a powerful tool in fields like accessibility, content generation, and image search.
## Introduction
The rapid advancement of deep learning in computer vision and natural language processing has fueled significant interest in automated image captioning, a challenging task that requires generating descriptive textual content from visual data. This interdisciplinary field combines insights from both image processing and language modeling, demanding robust and efficient models that can effectively bridge the gap between visual and textual information. A successful image captioning model not only requires precise object recognition but also a semantic understanding of scene relationships to generate human-like descriptions.

Two popular fusion strategies—early fusion and late fusion—offer distinct methods for integrating image and textual data in multimodal architectures. Early fusion involves combining visual and textual features at an initial stage, allowing subsequent layers to process the integrated information. In contrast, late fusion keeps each modality separate until the final stages, merging the outputs to generate captions. Both approaches have unique strengths and limitations, but their comparative effectiveness in image captioning remains an active area of exploration.

In this research, we systematically evaluate both early and late fusion techniques across multiple neural network architectures for visual and textual processing. For image encoding, we experiment with three state-of-the-art convolutional neural network (CNN) architectures—ResNet, Inception, and VGG—each known for its distinctive image feature extraction capabilities. These networks capture rich visual details and high-level features essential for understanding complex scenes. On the language modeling side, we use recurrent neural networks (RNNs), specifically Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), to handle the sequential nature of caption generation.

Our work aims to provide a comprehensive analysis of each permutation of fusion strategy, CNN backbone, and RNN-based language model. By evaluating the performance across all combinations, we seek to determine optimal configurations for enhanced image captioning. The findings from this study will offer valuable insights into effective fusion strategies and model choices, contributing to more efficient and accurate multimodal models for automated caption generation.

## Methodology
Our methodology explores early and late fusion strategies across multiple neural network architectures for image captioning, with experiments conducted on the Flickr8k dataset. This section details the dataset, preprocessing steps, feature extraction methods, model architectures, fusion strategies, and training process.
Dataset

The Flickr 8k dataset serves as the primary dataset, containing 8,000 images with five human-annotated captions per image. The diverse scenes in the dataset provide an opportunity to learn varied visual and textual associations. We preprocess the images by resizing and normalizing them, and tokenize and encode captions to standardize inputs.
## Image Feature Extraction
For image encoding, we experiment with three convolutional neural networks (CNNs):

* **ResNet:** Known for residual connections, allowing efficient deep training and providing detailed feature maps.

* **Inception:** Uses factorized convolutions to capture multiscale spatial hierarchies efficiently.

* **VGG:** With its uniform convolutional layers, VGG provides consistent and interpretable features.

All CNNs are pretrained on the ImageNet dataset, and we use the final convolutional layer outputs as input features for the fusion models.

## Text Feature Extraction
To generate sequential captions, we use recurrent neural networks (RNNs):

* **LSTM:** Known for its ability to capture long-term dependencies, useful in generating contextually consistent captions.
* **GRU:** A simpler alternative to LSTM, GRU provides computational efficiency while maintaining strong performance on sequential tasks.
  
The visual features are combined with the text features generated by either the LSTM or GRU to produce the final captioning model.
## Fusion Strategies
We explore two fusion methods:

* **Early Fusion:** Combines visual and textual features at the initial layers, allowing the model to learn a shared representation early on.
* **Late Fusion:** Keeps features separate until the final layer, allowing each modality to retain its unique properties until they are combined before caption generation.
  
With three CNNs, two RNNs, and two fusion strategies, we train 12 configurations.
## Training Procedure
Each model is trained in TensorFlow on Google Colab over 50 epochs. We use a cross-entropy loss function and the Adam optimizer with an initial learning rate of 1 X 10 -4 and a batch size of 32. Early stopping is implemented based on validation loss to prevent overfitting, with hyperparameter tuning applied to achieve optimal results across configurations.
## Experimental Setup and Tools
All models are implemented in TensorFlow on Google Colab, with training conducted on Colab's available GPUs. Results for each configuration are averaged over multiple trials to confirm stability and reduce random variation.



## Results
<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images/result_image.png" alt="" width="1000" height="500">

This image shows the loss graphs shared across the models. 


All models show a consistent decrease in loss over the epochs, indicating successful learning. The rate of decrease is more pronounced in the earlier epochs, which is typical for neural networks as they initially learn at a faster rate and then slow down as they converge. RESNET has better performance compared to VGG and INCEPTION and LATE FUSION gives better results than EARLY FUSION and LSTM outperforms GRU models. It is noted that LATE FUSION takes a longer time to train than EARLY FUSION due to passing of Images and Text features separately as compared to a concatenated pass.


<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images/Images/unnamed.png" alt="" width="300" height="200"> <img src="Images/dog_running_snow.png" alt="" width="300" height="200"> <img src="Images/dirt_bike.png" alt="" width="300" height="300">

<p align="center"><i>Results from RESNET model with LSTM in EARLY Fusion</i></p>
​
<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images/man_climbing_hill.png" alt="" width="300" height="200"><img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images/dog_jumping_bar.png" alt="" width="300" height="200"><img src="Images/dog_in_pond.png" alt="" width="300" height="200">

<p align="center"><i>Results from RESNET model with LSTM in LATE Fusion</i></p>

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/bike_air.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_on_grass.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/two_dogs_playing.png" alt="" width="300" height="200">

<p align="center"><i>Results from INCEPTION model with LSTM  in EARLY  Fusion</i></p>

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/children_playing_in_pool.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_running_through_grass.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_in_yard.png" alt="" width="300" height="200">

<p align="center"><i>Results from INCEPTION model with LSTM  in LATE Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/2_dogs_on_track.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/man_over_cliff.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/bike_on_track.png" alt="" width="300" height="200">

<p align="center"><i>Results from VGG model with LSTM  in EARLY Fusion</i></p>  

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_on_beach.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/man_skiing.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/man_with_dogs.png" alt="" width="300" height="200">

<p align="center"><i>Results from VGG model with LSTM  in LATE Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/girl_on_grass.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dogs_on_sand.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_bandw.png" alt="" width="300" height="200">

<p align="center"><i>Results from RESNET model with GRU  in EARLY Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/two_girls_posing.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/girl_on_swing.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dogs_playing.png" alt="" width="300" height="200">

<p align="center"><i>Results from RESNET model with GRU in LATE Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/girl_with_ball.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/man_red_shirt.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/man_and_woman.png" alt="" width="300" height="200">

<p align="center"><i>Results from INCEPTION model with GRU in LATE Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/girl_and_mother.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/dog_with_man.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/girl_in_black.png" alt="" width="300" height="200">
  
<p align="center"><i>Results from VGG model with GRU in LATE Fusion</i></p> 

<img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/men_on_bike.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/boy_in_yellow.png" alt="" width="300" height="200"> <img src="../imae_caption_AI/Analysis-of-Image-Captioning-Models/Images
/beach.png" alt="" width="300" height="200">
 
<p align="center"><i>Results from VGG model with GRU in EARLY Fusion</i></p> 

