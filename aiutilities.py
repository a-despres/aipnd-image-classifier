import json
import numpy as np
import os
import torch

from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms, models

def load_dataloaders(data_dir):
    ''' Defines the data directories, defines the image transforms, and loads the datasets.
    PARAMETERS:
        data_dir <String> - The specified location of the training, validation and testing data
    RETURNS:
        train_dataset <torch.utils.data.Dataset> - Dataset of training data
        train_dataloader <torch.utils.data.DataLoader> - The dataset and iterator to be used in training
        valid_dataloader <torch.utils.data.DataLoader> - The dataset and iterator to be used for validation
        test_dataloader <torch.utils.data.DataLoader> - The dataset and iterator to be used for testing
    '''
    
    # Data Directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Check if data directories exists
    if not os.path.isdir(data_dir) or not os.path.exists(data_dir):
        print('Data path "{}" does not exist or is not a directory.'.format(data_dir))
        print('Ending training sequence...')
        quit()
        
    if not os.path.isdir(train_dir) or not os.path.exists(train_dir):
        print('Training path: {}" does not exist or is not a directory.'.format(train_dir))
        print('Ending training sequence...')
        quit()
     
    if not os.path.isdir(valid_dir) or not os.path.exists(valid_dir):
        print('Validation path: {}" does not exist or is not a directory.'.format(valid_dir))
        print('Ending training sequence...')
        quit()
        
    if not os.path.isdir(test_dir) or not os.path.exists(test_dir):
        print('Testing path: {}" does not exist or is not a directory.'.format(test_dir))
        print('Ending training sequence...')
        quit()
    
    # Training Transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # Validation Transforms
    valid_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # Testing Transforms
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    return train_dataset, train_loader, valid_loader, test_loader

def select_active_device(use_gpu, is_training=False):
    ''' Determine what device should be used for training or inference (CPU or GPU).
    PARAMETERS:
        use_gpu <Bool> - GPU (True) or CPU (False)
        is_training <Bool> (Optional) - Training (True) or Inference (False)
    RETURN:
        device <String> - CPU (cpu) or GPU (cuda:0)
    '''
    # Prompt the user to use a GPU in training if not initially set as an option
    if torch.cuda.is_available() and use_gpu == False and is_training == True:
        print('\n')
        print('*' * 100)
        print('*** ALERT: It is recommended that a GPU be used during training.')
        print('*' * 100)
        print('\n')
        print('*** A GPU is available. Would you like to enable the GPU?')
        
        while True:
            user_response = input('*** Yes (Y) or No (N): ')
            
            if user_response.lower() == 'y' or user_response.lower() == 'yes':
                use_gpu = True
                break
            elif user_response.lower() == 'n' or user_response.lower() == 'no':
                use_gpu = False
                break
            else:
                print('Invalid input. Valid options are: Yes, Y, No, N')
                continue
    
    # Set the device to GPU if available
    if use_gpu == True and torch.cuda.is_available():
        device = 'cuda:0'
    
    # Alert the user that a GPU is unavailable if they included it as an option and set device to CPU
    elif use_gpu == True and not torch.cuda.is_available():
        print('\n')
        print('*' * 116)
        print('*** ALERT: A GPU is unavailable. A CPU will be used instead.')
        print('*' * 116)
        device = 'cpu'
        
    # Set device to CPU
    else:
        device = 'cpu'

    return device

def load_label_map(file):
    ''' Load a label map from a file stored on disk.
    PARAMETERS:
        file <String> - The specified location of the label map
    RETURN:
        cat_to_name <Dictionary> - A key, value list of categories
    '''
    # Quit application if label map does not exist
    if not os.path.exists(file) or os.path.isdir(file):
        print('"{}" does not exist.'.format(file))
        print('Ending inference sequence...')
        quit()
    
    # Open label map file
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

def resize_crop_image(image):
    ''' Resize and crop an input image. Returns the modified image.
    PARAMETERS:
        image <Image> - The original image to modify
    RETURN:
        c_image <Image> - The modified (resized and cropped) image
    '''
    # Original image dimensions
    width = image.size[0]
    height = image.size[1]
    
    crop_size = 224
    max_side = 256
    new_width = 0
    new_height = 0
    
    # Pixel Aspect Ratio
    par = width / height
    
    # Determine new dimensions of image, retaining aspect ratio
    if par < 1:
        new_width = 256
        new_height = int(par * max_side)
    else:
        new_width = int(par * max_side)
        new_height = 256
        
    new_dimensions = (new_width, new_height)
    
    # Resize image    
    image = image.resize(new_dimensions)
    
    # Crop image around center of image
    offset_w = new_width / 2
    offset_h = new_height / 2
    
    c_image = image.crop((
        offset_w - 112,
        offset_h - 112,
        offset_w + 112,
        offset_h + 112
    ))
    
    return c_image

def image_to_tensor(image):
    ''' Convert an Image into a tensor.
    PARAMETERS:
        image <Image> - The modified (resized and cropped) image
    RETURN:
        tensor <torch.Tensor> - A tensor representing the image
    '''
    # Constants used in normalization
    mean, standard_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Image to numpy array
    np_image = np.array(image) / 255
    
    # Normalize numpy array
    n_image = (np_image - mean) / standard_dev
    
    # Transpose dimensions of numpy array
    t_image = n_image.transpose(2, 0, 1)
    
    # Convert numpy array to tensor
    tensor = torch.from_numpy(t_image)

    return tensor

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.
    PARAMETERS:
        image_path <String> - The specified location of the image to open
    RETURN:
        tensor <torch.Tensor> - A tensor representing the image
    '''
    # Quit application if image does not exist
    if not os.path.exists(image_path) or os.path.isdir(image_path):
        print('"{}" does not exist.'.format(image_path))
        print('Ending inference sequence...')
        quit()
    
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    r_image = resize_crop_image(image)
    tensor = image_to_tensor(r_image)
    
    return tensor

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    PARAMETERS:
        image_path <String> - The specified location of the image to open
        model <Torchvision.Models> - The trained model
        device <String> - CPU (cpu) or GPU (cuda:0)
        topk <Int> (Optional) - The desired number of results to return
    RETURN:
        tensor <torch.Tensor> - A tensor representing the image
    '''
    
    # Prepare model for inference and send to CPU or GPU based on active device
    model.eval()
    model.to(device)
    torch.no_grad()
    
    # Prepare the image to send through model
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    # Send the image through the model
    output = model.double().forward(Variable(image))
    ps = torch.exp(output).data
    
    # TopK classes and probabilities
    probs, indices = ps.topk(topk)
    probs, indices = probs.cpu(), indices.cpu()
    
    # Invert class_to_idx dict
    idx_to_class = {i:c for c,i in model.class_to_idx.items()}
    
    classes = []    
    for i in indices.numpy()[0]:
        classes.append(idx_to_class[i])
    
    return probs.numpy()[0], classes