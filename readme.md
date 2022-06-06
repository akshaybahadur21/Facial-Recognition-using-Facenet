# Facial Recognition 🧔 🔍

[![](https://img.shields.io/github/license/sourcerer-io/hall-of-fame.svg?colorB=ff0000)](https://github.com/akshaybahadur21/Facial-Recognition-using-Facenet/blob/master/LICENSE.txt)  [![](https://img.shields.io/badge/Akshay-Bahadur-brightgreen.svg?colorB=ff0000)](https://akshaybahadur.com)

This code helps in facial recognition using facenets (https://arxiv.org/pdf/1503.03832.pdf). The concept of facenets was originally presented in a research paper.
The main concepts talked about triplet loss function to compare images of different person.
This concept uses inception network which has been taken from source and fr_utils.py is taken from deeplearning.ai for reference.
I have added several functionalities of my own for providing stability and better detection. 

## Code Requirements 🦄
You can install Conda for python which resolves all the dependencies for machine learning.

`pip install requirements.txt`

## Description 🕵️‍♂️
A facial recognition system is a technology capable of identifying or verifying a person from a digital image or a video frame from a video source. There are multiples methods in which facial recognition systems work, but in general, they work by comparing selected facial features from given image with faces within a database.

## Functionalities added 🧟
1) Detecting face only when your eyes are opened. (Security measure)
2) Using face align functionality from dlib to predict effectively while live streaming.


## Python  Implementation 👨‍🔬

1) Network Used- Inception Network
2) Original Paper - Facenet by Google

If you face any problem, kindly raise an issue

## File Organization 🗄️

```shell
├── Facial-Recognition-using-Facenet (Current Directory)
    ├── models : Saved Models
        ├── face-rec_Google.h5 : Facenet Model 
        └── shape_predictor_68_face_landmarks.dat : Facial Keypoints Model
    ├── utils : Utils Folder
        ├── fr_utils.py 
        └── inception_blocks_v2.py 
    ├── create_face.py : Store the faces for module
    ├── rec-feat.py - Main Application
    ├── Train-inception.py : Model Trainer
    ├── LICENSE
    ├── requirements.txt
    └── readme.md
        
```

## Setup 🖥️

1) If you want to train the network , run `Train-inception.py`, however you don't need to do that since I have already trained the model and saved it as 
`face-rec_Google.h5` file which gets loaded at runtime.
2) Now you need to have images in your database. The code check `/images` folder for that. You can either paste your pictures there or you can click it using web cam.
For doing that, run `create-face.py` the images get stored in `/incept` folder. You have to manually paste them in `/images folder`
3) Run `rec-feat.py` for running the application.


## Execution 🐉

```
python3 rec-feat.py
```

## Results 📊

<img src="https://github.com/akshaybahadur21/BLOB/blob/master/Face-Rec.gif">

###### Made with ❤️ and 🦙 by Akshay Bahadur

## References 🔱
 
 - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
 - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
 - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
 - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet  






