import argparse
import aiutilities as util
import aimodel as ai

def define_arg_parser():
    ''' Define the command line arguments
    PARAMETERS:
        None
    RETURN:
        parser <argparse.ArgumentParser> - The configured parser with all command line arguments.
    '''
    parser = argparse.ArgumentParser()

    # Argument: Data Directory
    parser.add_argument('path_to_image',
                        help='The path to the image you wish to run through the classification prediction network.')

    # Argument: Checkpoint
    parser.add_argument('checkpoint',
                        type=str,
                        help='The path to the checkpoint you wish to use with the network.')

    # Argument: Top-K
    parser.add_argument('-k', '--top_k',
                        type=int,
                        default=5,
                        help='List the top-most predictions for the processed image. The default value is 5.')

    # Argument: Category Names
    parser.add_argument('-c', '--category_names',
                        type=str,
                        default='cat_to_name.json',
                        help='The path to a custom list of category names.')

    # Argument: Use GPU (BOOL)
    parser.add_argument('-g', '--gpu',
                        dest='gpu',
                        action='store_true',
                        help='Toggle the use of the GPU for inference (if available).')

    return parser

def display_prediction_overview(in_args, device, architecture, top_k):
    ''' A summary of the prediction parameters.
    PARAMETERS:
        in_args <argparse.Namespace> - A key, value list of arguments supplied from the command line
        device <String> - CPU (cpu) or GPU (cuda:0)
        architecture <String> - The architecture of the model being used
        top_k <Int> - The number of results to display
    RETURN:
        None
    '''
    device_text = 'GPU' if device == 'cuda:0' else 'CPU'
    arch_text = 'Densenet-121' if architecture == 'densenet121' else 'VGG-16'

    print('\n')
    print('*' * 100)
    print('*** Running classifier for "{}" on: {}'.format(in_args.path_to_image, device_text))
    print('*** Using architecture: {}'.format(arch_text))
    print('*** Top-K: {} | Label Map: {}'.format(top_k, in_args.category_names))
    print('*' * 100)

def display_top_results(probs, classes, label_map, top_k):
    ''' A listing of top results returned by the model.
    PARAMETERS:
        probs <numpy.ndarray> - List of probabilities
        classes <List> - List of classes
        label_map <Dictionary> - A key, value list of categories
        top_k <Int> - The number of results to display
    RETURN:
        None
    '''
    top_class = label_map["{}".format(classes[0])].title()
    top_prob = probs[0] * 100

    print('\n')
    print('*** Top {} Results'.format(top_k))
    for i in range(top_k):
        prob_str = str('{:.2f}'.format(probs[i] * 100)).zfill(5)
        class_label = label_map['{}'.format(classes[i])].title()
        print('*** {}% - {}'.format(prob_str, class_label))
    print('\n')

def main():
    parser = define_arg_parser()
    in_args = parser.parse_args()

    device = util.select_active_device(in_args.gpu)
    label_map = util.load_label_map(in_args.category_names)
    model, architecture = ai.load_checkpoint(in_args.checkpoint)

    image = util.process_image(in_args.path_to_image)

    top_k = 5 if in_args.top_k == 0 else in_args.top_k
    probs, classes = util.predict(in_args.path_to_image, model, device, top_k)

    display_prediction_overview(in_args, device, architecture, top_k)
    display_top_results(probs, classes, label_map, top_k)

if __name__ == '__main__':
    main()
