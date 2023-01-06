2023-DTU-MLOps-Object-Detection
==============================

This repository contains the project work carried out as a final project in Machine Learning Operations course at DTU ([course website](https://kurser.dtu.dk/course/02476)).

**Group 25** \
Ditte Jepsen \
Agata Makarewicz \
Amanda Sommer \
Jacek Wiśniewski \
(see contributors list for individual github pages).

## Project Description
### Overall goal of the project

The goal of the project is to perform an object detection task using the DEtection TRansformer (DETR) model to locate instances of objects of different classes within an image. 

### Framework

We will be using the [Transformers](https://github.com/huggingface/transformers) framework, which provides thousands of pretrained models to perform tasks on text, vision, and audio.

### Usage of the selected framework in the project

We are going to start with a pre-trained DETR model from [Huggingface](https://huggingface.co/facebook/detr-resnet-50) available through the Transformers framework, originally introduced in the paper [End-to-End Object Detection with Transformers by Carion et al.](https://arxiv.org/abs/2005.12872). Then, we intend to make use of the multiple image processors and pre-trained models which the framework offers, looking for opportunities to improve.

### Data

Primarly, we will use the [COCO 2017 dataset](https://cocodataset.org/#download) which is a is a large-scale object detection dataset with over 200K labeled images and over 150 categories. It is also the dataset on which the pre-trained model we will use was trained on. After that, we will try other benchmark datasets for object detection task from [roboflow](https://public.roboflow.com/object-detection) like [Pascal VOC 2012](https://public.roboflow.com/object-detection/pascal-voc-2012) or [Mask Wearing Dataset](https://public.roboflow.com/object-detection/mask-wearing).

### Deep Learning models

We will use the [DEtection TRansformer (DETR)](https://github.com/facebookresearch/detr) - a model using an encoder-decoder Transformer on top of a convolutional backbone. It uses a conventional CNN backbone to learn a 2D representation of an input image. The model we will start with uses a ResNet-50 as a convolutional backbone. Due to limited time and computational resources, we will use the pre-trained models with different datasets, and might try to improve them with fine-tuning.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
