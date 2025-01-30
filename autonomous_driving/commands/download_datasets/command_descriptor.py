"""Represents a module that contains the descriptor for the command for downloading the cifar10 dataset."""

from argparse import ArgumentParser

from autonomous_driving.commands.base import BaseCommandDescriptor
from autonomous_driving.src.datasets import DATASET_IDS, DEFAULT_DATASET_ID


class DownloadDatasetsCommandDescriptor(BaseCommandDescriptor):
    """Represents the description of the download dataset command."""

    def get_name(self) -> str:
        """Gets the name of the command.

        Returns:
            str: Returns the name of the command.
        """

        return 'download-datasets'

    def get_description(self) -> str:
        """Gets the description of the command.

        Returns:
            str: Returns the description of the command.
        """

        return '''Downloads the specified torchvision dataset'''

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Adds the command line arguments to the command line argument parser.

        Args:
            parser (ArgumentParser): The command line argument parser to which the arguments are to be added.
        """
        parser.add_argument(
            'dataset_path',
            type=str,
            help='The path to the directory into which the specified dataset is retrieved or downloaded.'
        )
        parser.add_argument(
            'dataset',
            type=str,
            default=DEFAULT_DATASET_ID,
            choices=DATASET_IDS,
            help=f'Name of dataset used in the federated training process. Defaults to "{DEFAULT_DATASET_ID}".'
        )
        parser.add_argument(
            '-s'
            'split_ratio',
            type=float,
            default=0.2,
            help='The ratio of data of training set that is utilized for validation.'
        )
