2023-DTU-MLOps-Object-Detection
==============================

Final project in Machine Learning Operations course at DTU.

## Overall goal of the project


The goal of the project is to use the DEtection TRansformer (DETR) model to detect objects in the images.

## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)

Since we chose an image-related problem, we are going to use the Transformers framework, which provides thousands of pretrained models to perform tasks on text, vision, and audio.

## How do you intend to include the framework in your project

We are going to start with testing a pre-trained Res-Net model from huggingface (https://huggingface.co/facebook/detr-resnet-50) on the training dataset - COCO 2017. After that, we will test the model on other datasets, and we will look for opportunities to improve the model.

## What data are you going to run on

Firstly, we will test the model on a COCO 2017 dataset (https://cocodataset.org/#download) which is a training dataset for the model we will work with. After that, we will try other benchmark datasets for object detection from roboflow (https://public.roboflow.com/object-detection) like Pascal VOC 2012 (https://public.roboflow.com/object-detection/pascal-voc-2012) or Mask Wearing Dataset (https://public.roboflow.com/object-detection/mask-wearing).

## What deep learning models do you expect to use

We will use the pre-trained model that uses DEtection TRansformer (DETR) which is an encoder-decoder transformer. It uses ResNet-50 as a convolutional backbone. For the sake of lack of time, we will focus on testing the pre-trained model with different datasets, and try to improve it with fine-tuning, to reduce the time needed for training.







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
