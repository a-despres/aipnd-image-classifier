# AI Programming with Python Nanodegree - Final Project - Image Classifier
A neural network that takes can be trained for classifying images.

This is the final project for Udacity's AI Programming with Python Nanodegree.

## Installation
Close this repository to your desktop.
```sh
$ git clone https://github.com/a-despres/aipnd-image-classifier.git
$ cd aipnd-image-classifier
```

---
## Training the Network
Before inference can be done the network must be trained.
  - Download an image dataset for training, such as [Flowers](https://drive.google.com/open?id=1eDKmf64ozfkkMlJ-uv1bsKjd1o_H_VZK) (ZIP, 349.5 MB). This dataset contains 3 sets of images: train, test and valid, as well as a JSON file for converting image categories into names. Training the network does not require this specific dataset, but the dataset should be organized in the same way.

  - After you have a dataset you will use `train.py` to train the network.

**Due to the complexity and number of calculations being performed by the neural network, the use of a GPU when training is highly recommended.**

## Usage: train.py

#### Required Arguments
  - `data_dir` - The name of the directory containing the "train", "valid" and "test" data directories.

#### Optional Arguments
  - `-a` or `--arch` - The model architecture. Options are: "vgg16" or "densenet121"; "vgg16" is selected by default.
  - `-e` or `--epochs` - The number of training epochs. The default is 3.
  - `-g` or `--gpu` - Toggle the use of the GPU when training (if available).
  - `-h` or `--hidden_units` - The number of hidden units, or nodes, contained in the hidden layer. The default is 512.
  - `-l` or `--learning_rate` - The learning rate used in training the network. The default rate is 0.001.
  - `-s` or `--save-dir` - The directory to be used when saving network checkpoints.

#### Example
```sh
$ python train.py flowers -a densenet121 -e 5 -h 450 -l 0.002 -s checkpoints -g
```

---
## Inference
After the network has been trained it can then be used for inference. You will use `predict.py` to predict the classification of any image.

## Usage: predict.py

#### Required Arguments
  - `path_to_image` - The path to the image you wish to run through the network.
  - `checkpoint` - The path to the network checkpoint that was saved when training was completed.

#### Optional Arguments
  - `-c` or `--category_names` - The path to a custom list of category names. The default value is "cat_to_name.json."
  - `-g` or `--gpu` - Toggle the use of the GPU for inference (if available).
  - `-k` or `--top_k` - List the top-most predictions for the selected image. The default value is 5.

#### Example
```sh
$ python predict.py images/unknown_flower.jpg checkpoints/densenet121.pth -c cat_to_name_alt.json -k 10 -g
```
