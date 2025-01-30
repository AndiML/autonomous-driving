"""Represents a module that contains the vanilla federated averaging command."""

import logging
from argparse import Namespace
from importlib import import_module
from inspect import getmembers, isclass

from autonomous_driving.commands.base import BaseCommand
from autonomous_driving.src.datasets import DATASET_IDS
from autonomous_driving.src.datasets.dataset import Dataset


class DownloadDatasetsCommand(BaseCommand):
    """Represents a command that command for the download of datasets."""

    def __init__(self) -> None:
        """Initializes a new FederatedAveraging instance. """

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Downloads the specified dataset
        self.logger.info("Downloading %s Dataset", command_line_arguments.dataset.upper())

        if command_line_arguments.dataset in DATASET_IDS:
            # Loads the class corresponding to the specified dataset
            dataset_module = import_module("autonomous_driving.src.datasets")
            dataset_module_classes = getmembers(dataset_module, isclass)
            for _, class_object in dataset_module_classes:
                if Dataset in class_object.__bases__ and hasattr(class_object, 'dataset_id') \
                        and getattr(class_object, 'dataset_id') == command_line_arguments.dataset:

                    dataset_class = class_object
                    break

            kitti_dataset = dataset_class(command_line_arguments.dataset_path)
        else:
            exit("Dataset not supported")
