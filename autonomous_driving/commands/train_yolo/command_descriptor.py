"""Represents a module that contains the descriptor for the federated-learning-baseline command."""

from argparse import ArgumentParser

from autonomous_driving.commands.base import BaseCommandDescriptor
from autonomous_driving.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID


class TrainYoloCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of federated averaging algorithm command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'train-yolo'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Evaluates the baseline performance of the model.'''

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """

        parser.add_argument(
            'output_path',
            type=str,
            help='The path to the directory into which the results of the experiments are saved.'
        )
        # Path to dataset
        parser.add_argument(
            'dataset_path',
            type=str,
            help='Directory in whcih dataset is stored..'
        )

        parser.add_argument(
            '-d',
            '--dataset',
            type=str,
            default=DEFAULT_DATASET_ID,
            choices=DATASET_IDS,
            help=f'Name of dataset used for training. Defaults to "{DEFAULT_DATASET_ID}".'
        )

        parser.add_argument(
            '-s'
            '--split_ratio',
            type=float,
            default=0.8,
            help='The ratio of data of training set that is utilized for validation.'
        )

        # Training hyperparameters
        parser.add_argument(
            '-e',
            '--number_of_epochs',
            type=int,
            default=100,
            help="Number of training epochs."
        )
        parser.add_argument(
            '-b',
            '--batchsize',
            type=int,
            default=10,
            help="Batch size for YOLO training."
        )
        parser.add_argument(
            '-l',
            '--learning_rate',
            type=float,
            default=0.01,
            help='Initial learning rate (lr0) for YOLO.'
        )
        parser.add_argument(
            '-W',
            '--weight_decay',
            type=float,
            default=0.0005,
            help='Weight decay for YOLO.'
        )
        parser.add_argument(
            '-m',
            '--set_momentum',
            type=float,
            default=0.937,
            help='Momentum used by SGD (ignored if using Adam).'
        )
        parser.add_argument(
            '-o',
            '--optimizer',
            type=str,
            default='sgd',
            choices=['sgd', 'adam'],
            help="Optimizer choice for YOLO."
        )

        # YOLO-specific arguments
        parser.add_argument(
            '-i',
            '--image_size',
            type=int,
            default=640,
            help="Input image size for training."
        )

        # Optional: If you want to specify a YOLO model or config explicitly
        parser.add_argument(
            '-p',
            '--model_path',
            type=str,
            default=None,
            help="Path to YOLO model checkpoint."
        )

        # Optional: If you plan to override the default trainer
        parser.add_argument(
            '-c',
            '--use_costum_loss_function',
            action='store_true',
            help='Whether to use a ustom trainer script that overrides YOLO loss function.'
        )

        parser.add_argument(
            'C',
            '--customize_dataset',
            action='store_true',
            help="Enable custom dataset processing, including user-defined augmentations and data loading."
        )

        parser.add_argument(
            '-g',
            '--use_gpu',
            action='store_true',
            help="Use CUDA for training if available."
        )
