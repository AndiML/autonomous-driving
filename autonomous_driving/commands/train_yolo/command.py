"""Represents a module that contains the training command adapted for YOLO model."""

import logging
import os

from argparse import Namespace
from importlib import import_module
from inspect import getmembers, isclass

from autonomous_driving.commands.base import BaseCommand
from autonomous_driving.src.datasets.dataset import Dataset
from autonomous_driving.src.datasets.data_generator import create_dataloader

from ultralytics import YOLO

class TrainYoloCommand(BaseCommand):
    """Represents a command that represents the training command using YOLO model."""

    def __init__(self) -> None:
        """Initializes a new TrainYoloCommand instance."""
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def run(self, command_line_arguments: Namespace) -> None:
        """Runs the command using YOLOv8 for training.

        Args:
            command_line_arguments (Namespace): The parsed command line arguments.
        """
        # Select device for training
        device = 'cuda' if command_line_arguments.use_gpu else 'cpu'
        self.logger.info('Selected %s for Training Process', device.upper())

        # Download the dataset and retrieve config YAML of specfied dataset
        self.logger.info(f'Download and Retrieving YAML File for {command_line_arguments.dataset.upper()}', extra={'start_section': True})

        # Loads the class corresponding to the specified dataset
        dataset_module = import_module("autonomous_driving.src.datasets")
        dataset_module_classes = getmembers(dataset_module, isclass)
        for _, class_object in dataset_module_classes:
            if Dataset in class_object.__bases__ and hasattr(class_object, 'dataset_id') \
                    and getattr(class_object, 'dataset_id') == command_line_arguments.dataset:

                dataset_class = class_object
                break

            kitti_dataset = dataset_class(
                path = command_line_arguments.dataset_path,
                split_ratio = command_line_arguments.split_ratio,
                customize_dataset = command_line_arguments.customize_dataset
            )
            kitti_dataset.download()
        else:
            exit("Dataset not supported")


        dataset_path = command_line_arguments.dataset_path
        yaml_config_path = os.path.join(dataset_path, f"{command_line_arguments.dataset}.yaml")
        if not os.path.exists(yaml_config_path):
            raise FileNotFoundError(f"YAML configuration file not found at {yaml_config_path}.")


        # Loads YOLO model
        if command_line_arguments.model_path is not None:
            model = YOLO(command_line_arguments.model_path)
        else:
            model = YOLO()  # starts with no weights/config

        if command_line_arguments.use_costum_loss_function:
            print('Not implemented')
            # Dynamically load custom
            # from your_package.my_trainer import MyCustomTrainer
            # model.trainer = MyCustomTrainer(model=model)

        # Train parameters
        epochs = command_line_arguments.number_of_epochs
        batch_size = command_line_arguments.batchsize
        lr0 = command_line_arguments.learning_rate
        imgsz = command_line_arguments.image_size
        momentum = command_line_arguments.set_momentum
        optimizer = command_line_arguments.optimizer
        weight_decay = command_line_arguments.weight_decay

        # Start training
        model.train(
            data=yaml_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            lr0=lr0,
            momentum=momentum,
            optimizer=optimizer,
            weight_decay=weight_decay,
            device=device,
            project=command_line_arguments.output_path,
            name="yolo_baseline",
            exist_ok=True,
            verbose=True,
        )

        self.logger.info('YOLO Training Completed Successfully!')

