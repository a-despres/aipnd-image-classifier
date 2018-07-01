import aimodel as ai
import aiutilities as util
import argparse
import os

def define_arg_parser():
    ''' Define the command line arguments
    PARAMETERS:
        None
    RETURN:
        parser <argparse.ArgumentParser> - The configured parser with all command line arguments.
    ''' 
    parser = argparse.ArgumentParser()
    
    # Argument: Data Directory
    parser.add_argument('data_dir',
                        help='The name of the directory containing the "train", "valid" and "test" data directories.')
    
    # Argument: Save Directory
    parser.add_argument('-s', '--save_dir',
                        type=str,
                        default=os.getcwd(),
                        help='The directory used when saving network checkpoints.')
    
    # Argument: Architecture
    parser.add_argument('-a', '--arch',
                        type=str,
                        default='vgg16',
                        help='Choose the model architecture. Options are: "vgg16" or "densenet121"; "vgg16" is selected by default.')
    
    # Argument: Hyperparameter - Learning Rate
    parser.add_argument('-l', '--learning_rate',
                        type=float,
                        default=0.001,
                        help='The learning rate used in training the network. The default rate is 0.001.')
    
    # Argument: Hyperparameter - Hidden Units
    parser.add_argument('-u', '--hidden_units',
                        type=int,
                        default=512,
                        help='The number of hidden units in the hidden layer of the network model. The default number of hidden units is 512.')
    
    # Argument: Hyperparameter - Epochs
    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=3,
                        help='The number of epochs used in training the network. The default number of epochs is 3.')
    
    # Argument: Use GPU (BOOL)
    parser.add_argument('-g', '--gpu',
                        dest='gpu',
                        action='store_true',
                        help='Toggle the use of the GPU when training (if available).')
    
    return parser

def validate_args(in_args):
    ''' Make sure the hyperparameter arguments are valid.
    PARAMETER:
        in_args <argparse.Namespace> - A key, value list of arguments supplied from the command line
    RETURN:
        in_args <argparse.Namespace> - A modified key, value list of arguments supplied from the command line
    '''
    # Make sure epochs is greater than 0. If 0, set to 3 (Default)
    in_args.epochs = 3 if in_args.epochs <= 0 else in_args.epochs
    
    # Make sure learning rate is greater than 0, if 0, set to 0.001 (Default)
    in_args.learning_rate = 0.001 if in_args.learning_rate <= 0 else in_args.learning_rate
    
    # Make sure hidden units if greater than 0, if 0, set to 512 (Default)
    in_args.hidden_units = 512 if in_args.hidden_units <= 0 else in_args.hidden_units
    
    return in_args

def display_training_overview(in_args, device):
    ''' A summary of the training parameters.
    PARAMETERS:
        in_args <argparse.Namespace> - A key, value list of arguments supplied from the command line
        device <String> - CPU (cpu) or GPU (cuda:0)
    RETURN:
        None
    '''  
    device_text = 'GPU' if device == 'cuda:0' else 'CPU'
    arch_text = 'Densenet-121' if in_args.arch == 'densenet121' else 'VGG-16'
    
    print('\n')
    print('*' * 100)
    print('*** Training model with dataset "{}" on: {}'.format(in_args.data_dir, device_text))
    print('*** Using architecture: {}'.format(arch_text))
    print('*** Learning Rate: {} | Hidden Unit: {} | Epochs: {}'.format(in_args.learning_rate, in_args.hidden_units, in_args.epochs))
    print('*' * 100)
    print('\n')

def main():
    parser = define_arg_parser()
    in_args = parser.parse_args()
    in_args = validate_args(in_args)
    
    train_dataset, train_loader, valid_loader, test_loader = util.load_dataloaders(in_args.data_dir)
    
    model = ai.load_pretrained_model(in_args.arch)
    model = ai.generate_and_assign_classifier(model, in_args.arch, in_args.hidden_units)
    
    device = util.select_active_device(in_args.gpu, True)
    display_training_overview(in_args, device)
        
    model = ai.train_model(model, device, in_args.epochs, in_args.learning_rate, train_loader, valid_loader)
    ai.test_model(model, device, test_loader)
    ai.save_checkpoint(model, train_dataset, in_args.arch, in_args.save_dir)
    
if __name__ == '__main__':
    main()