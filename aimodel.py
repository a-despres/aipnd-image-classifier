import os
import sys
import torch
import torch.nn.functional as F

from collections import OrderedDict
from time import sleep
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

def load_pretrained_model(arch):
    ''' Load a pretrained model based on user input.
    PARAMETER:
        arch <String> - Desired network architecture.
    RETURN:
        model <Torchvision.Models> - Pretrained model.
    '''
    # Densenet 121
    if arch == 'densenet121': # yes
        model = models.densenet121(pretrained=True)
    # VGG16 (Default)
    else:
        model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def generate_and_assign_classifier(model, arch, hidden_units):
    ''' Create a new classifier to be used with a pretrained model.
    PARAMETERS:
        model <Torchvision.Models> - Pretrained model
        arch <String> - Model architecture
        hidden_units <Int> - Desired number of nodes in hidden layer
    RETURN:
        model <Torchvision.Models> - Pretrained model with new classifier
    '''
    #Densenet 121
    if arch == 'densenet121':
        input_units = 1024
    # VGG16 (Default)
    else:
        input_units = 25088
        
    new_ff_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
        
    model.classifier = new_ff_classifier
    return model

def train_model(model, device, epochs, learning_rate, train_loader, valid_loader):
    ''' Train a predefined model with user-defined hyperparameters: epochs and learning rate.
    PARAMETERS:
        model <Torchvision.Models> - Pretrained model with classifier
        device <String> - CPU (cpu) or GPU (cuda:0)
        epochs <Int> - The desired number of epochs to be used in training
        learning_rate <Float> - The desired learned rate to be used in training
        train_loader <torch.utils.data.DataLoader> - The dataset and iterator to be used in training
        valid_loader <torch.utils.data.DataLoader> - The dataset and iterator to be used in validation
    RETURN:
        model <Torchvision.Models> - Trained model
    '''    
    # Train the model with the new classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    print_every = 40
    steps = 0
    running_loss = 0
    
    print('*** Training model...')
    
    for e in range(epochs):
        model.train()

        for images, labels in iter(train_loader):
            steps += 1

            inputs, targets = Variable(images), Variable(labels)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            train_output = model.forward(inputs)
            train_loss = criterion(train_output, targets)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

            if steps % print_every == 0:
                model.eval()

                accuracy = 0
                valid_loss = 0

                for ii, (images, labels) in enumerate(valid_loader):
                    inputs, labels = Variable(images), Variable(labels)
                    inputs, labels = inputs.to(device), labels.to(device)

                    torch.no_grad()
                    valid_output = model.forward(inputs)
                    valid_loss = criterion(valid_output, labels)

                    running_loss += valid_loss.item()

                    probs = torch.exp(valid_output).data
                    equality = (labels.data == probs.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("*** Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.3f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.3f} | ".format(valid_loss/len(valid_loader)),
                     "Validation Accuracy: {:.2f}%".format(accuracy/len(valid_loader) * 100))

                running_loss = 0
                model.train()
    print('*** Training complete.\n')
    return model

def test_model(model, device, test_loader):
    ''' Run the test images through the model in inference mode. Each image is matched with a label to determine accuracy.
    PARAMETERS:
        model <Torchvision.Models> - Trained model
        device <String> - CPU (cpu) or GPU (cuda:0)
        test_loader <torch.utils.data.DataLoader> - The dataset and iterator to be used in testing
    RETURN:
        None
    '''
    # Set model for inference
    model.eval()
    
    # Set the number of correct image matchs and total images to zero (0)
    correct, total = 0, 0
    
    print('*' * 100)
    print('*** Testing model for accuracy...')
    
    # Run batches of images through the network and write status to terminal window.
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            probs, prediction = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
            # Output status to command line
            sys.stdout.write('\r')
            sys.stdout.write('*** Tested {} of {} images'.format(total, len(test_loader.dataset)))
            sys.stdout.flush()
            sleep(0.25)

    print('\n*** Model accuracy: {:.2f}%'.format(100 * correct / total))
    print('*' * 100)

def save_checkpoint(model, dataset, arch, save_dir):
    ''' Save the model checkpoint to disk.
    PARAMETERS:
        model <Torchvision.Models> - Trained model
        arch <String> - CPU (cpu) or GPU (cuda:0)
        save_dir <String> - The desired location to save the checkpoint
    RETURN:
        None
    '''
    # Define the appropriate number of input nodes based on architecture and the filename for saving.
    input_size = model.classifier[0].in_features
    
    # Densenet 121
    if arch == 'densenet121':
        checkpoint_filename = 'densenet121.pth'
    # VGG16 (Default)
    else:
        checkpoint_filename = 'vgg16.pth'
    
    print('\n')
    print('*** Saving model checkpoint as: "{}"...'.format(checkpoint_filename))
    
    # check if save directory exists, if not create it.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # create checkpoint and save it to disk
    model.cpu()
    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
        'architecture': arch,
        'input_size': input_size,
        'output_size': 102,
        'image_dataset': model.class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, '{}/{}'.format(save_dir, checkpoint_filename))

    print('*** Model checkpoint saved to: {}.'.format(save_dir))
    print('\n')
    
def load_checkpoint(file):
    ''' Load a checkpoint from file and return the appropriate model and a string identifying the model architecture.
    PARAMETERS:
        file <String> - The specified location of the checkpoint
    RETURNS:
        model <Torchvision.Models> - Trained model
        arch <String> - Model architecture 
    '''
    # Determine if checkpoint file exists; quit if file does not exist
    if not os.path.exists(file) or os.path.isdir(file):
        print('Checkpoint "{}" does not exist.'.format(file))
        print('Ending inference sequence...')
        quit()
    
    # Load the checkpoint from file
    checkpoint = torch.load(file)
    
    # The type of architecture is being used for the loaded checkpoint
    arch = checkpoint['architecture']
    
    # Load the appropriate pre-trained model
    model = load_pretrained_model(arch)
    
    # Assign values from the checkpoint to the model
    model.class_to_idx = checkpoint['image_dataset']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, arch