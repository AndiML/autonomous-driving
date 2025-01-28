"""Represents a module containing the creation of the data for the federated learning process."""

from importlib import import_module
from inspect import getmembers, isclass
from typing import Any

from autonomous_driving.src.datasets.dataset import Dataset


def create_dataset(
    dataset_path: str,
    dataset_kind: str
) -> tuple[Dataset, dict[int, dict[Any, Any]]]:
    """Creates the datasets for each client according to the specified partition strategy

    Args:
        dataset_path (str): The path were the dataset is stored. If the dataset could not be found, then it is automatically downloaded to the
            specified location.
        number_of_clients (int): Number of clients to partition the dataset.
        dataset_kind (str): The kind of dataset that is used for local training.
    Raises:
        ValueError: If the sub-class that implements the abstract base class for datasets did not specify a sample shape, an exception is raised.

    Returns:
        tuple[Dataset, dict[int, dict[Any, Any]]]: A list that contains the training data for each client and the central server
            validation dataset. Furthermore it returns the instance of the data class used for model generation.
    """
    # Loads the class corresponding to the specified dataset
    dataset_module = import_module("autonomous_driving.src.datasets")
    dataset_module_classes = getmembers(dataset_module, isclass)
    dataset_class: type | None = None
    for _, class_object in dataset_module_classes:
        if Dataset in class_object.__bases__ and hasattr(class_object, 'dataset_id') and getattr(class_object, 'dataset_id') == dataset_kind:
            dataset_class = class_object
            break
    if dataset_class is None:
        raise ValueError(f'No dataset of the specified kind "{dataset_kind}" could be found.')
    data_class_instance: Dataset = dataset_class(dataset_path)


    return data_class_instance.get_training_data_loader(batch_size=10, shuffle_samples=True)


