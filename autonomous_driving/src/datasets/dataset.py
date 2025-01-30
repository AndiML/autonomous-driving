"""A module that contains the abstract base class for all datasets."""

import random
from typing import Any
from abc import ABC, abstractmethod
from typing_extensions import Protocol

import numpy
import torch


class DatasetData(Protocol):
    """Represents protocol for the data of a dataset, from which samples of the dataset can be retrieved."""

    def __len__(self) -> int:
        """Gets the number of samples in the data.

        Raises:
            NotImplementedError: Since this is a protocol, the method is not implemented and always raises a NotImplementedError.

        Returns:
            int: Returns the number of samples in the data.
        """

        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """Gets the sample with the specified index.

        Args:
            index (int): The index of the samples that is to be retrieved.

        Raises:
            NotImplementedError: Since this is a protocol, the method is not implemented and always raises a NotImplementedError.

        Returns:
            Any: Returns the retrieved sample.
        """

        raise NotImplementedError


class DatasetSplit(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """An abstract Dataset class wrapped around Pytorch Dataset class used for slicing."""

    def __init__(self, dataset: DatasetData, indices: list[int]):
        """Initializes the a DatasetSplit instance.

        Args:
            dataset (DatasetData): The dataset which needs to be splitted.
            indices (list[int]): The list of indices that indexes the data of the dataset.
        """
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.tensor, torch.tensor]:
        image, label = self.dataset[self.indices[item]]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            # If it's a tensor, ensure it's detached
            label = label.detach()
        return image.clone().detach(), label


class Dataset(ABC):
    """Represents the abstract base class for all datasets."""

    def __init__(self, labels: list[int]) -> None:
        """Initializes a Dataset instance.

        Args:
            labels (list[int]): The list of labels for all training samples.
        """
        self.distinct_classes = len(numpy.unique(labels))
        self._training_data: list[DatasetSplit] = []
        self._validation_data: list[DatasetSplit] = []


    @property
    @abstractmethod
    def training_data(self) -> DatasetData:
        """Gets the training data of the dataset.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.

        Returns:
            DatasetData: Returns the training data of the dataset.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def validation_data(self) -> DatasetData:
        """Gets the validation data of the dataset.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.

        Returns:
            DatasetData: Returns the validation data of the dataset.
        """

        raise NotImplementedError


    @property
    @abstractmethod
    def sample_shape(self) -> tuple[int, ...]:
        """Gets the the shape of the samples.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.

        Returns:
            tuple[int, ...]: Returns a tuple that contains the sizes of all dimensions of the samples.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_classes(self) -> int:
        """Gets the number of distinct classes.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.

        Returns:
            int: Returns the number of distinct classes.
        """

        raise NotImplementedError

    @abstractmethod
    def get_labels(self) -> list[int]:
        """Retrieves the labels of the dataset.

        Raises:
            NotImplementedError: As this is an abstract method that must be implemented in sub-classes, a NotImplementedError is always raised.

        Returns:
            list[int]: Returns a list of the labels.
        """

        raise NotImplementedError

    @property
    def training_data(self) -> list[DatasetSplit]:
        """Gets the training data of the dataset.

        Returns:
            list[DatasetSplit]: Returns the training data of the dataset.
        """

        return self._training_data

    @property
    def validation_data(self) -> list[DatasetSplit]:
        """Gets the validation data of the dataset.

        Returns:
            list[DatasetSplit]: Returns the validation data of the dataset.
        """

        return self._validation_data


    def get_training_data_loader(self, batch_size: int, shuffle_samples: bool) -> torch.utils.data.DataLoader:
        """Creates a data loader for the training data of the dataset.

        Args:
            batch_size (int): The number of samples per batch.
            shuffle_samples (bool): Determines whether the samples of the dataset should be in their original order or in a random order.

        Returns:
            torch.utils.data.DataLoader: Returns the created data loader.

        """

        return create_data_loader(self.training_data, batch_size, shuffle_samples=shuffle_samples)


    def get_validation_data_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Creates a data loader for the validation data of the dataset.

        Args:
            batch_size (int): The number of samples per batch.

        Returns:
            torch.utils.data.DataLoader: Returns the created data loader.
        """

        return create_data_loader(self.validation_data, batch_size, shuffle_samples=False)


def create_data_loader(data: DatasetData | DatasetSplit, batch_size: int, shuffle_samples: bool) -> torch.utils.data.DataLoader:
    """Creates a data loader for the specified dataset data.

    Args:
        data (DatasetData | DatasetSplit): The dataset data for which the data loader is to be created.
        batch_size (int): The number of samples per batch.
        shuffle_samples (bool): Determines whether the samples of the dataset should be in their original order or in a random order.

    Raises:
        ValueError: If the specified dataset data is not a subclass of PyTorch's Dataset class, an exception is raised. In principle, dataset data can
            be of any type as long as it implements the DatasetData protocol, i.e., the __len__ and __getitem__ methods, but PyTorch's DataLoader
            class only excepts "real" PyTorch datasets.

    Returns:
        torch.utils.data.DataLoader: Returns the created data loader.
    """

    # We are using the DatasetData protocol for the training and validation data to hide the fact that internally they are PyTorch datasets, in
    # principle, the training and validation data could also be of another type, as long as they implement the DatasetData protocol, since the PyTorch
    # DataLoader only supports "real" PyTorch datasets, we have make sure that the training and validation data is actually a subclass of PyTorch's
    # Dataset class (the added benefit is, that MyPy now also knows that this is a PyTorch Dataset and does not complain when we pass it to the
    # constructor of the DataLoader)
    #if not isinstance(data, torch.utils.data.Dataset):
    #    raise ValueError('Data loaders can only be created for PyTorch datasets.')

    # Creates a generator, which manages the state of a PyTorch's pseudo random number generator, this is passed to the data loader in order to set
    # the random seed on the worker processes that are used to actually load the dataset samples
    random_number_generator_state_manager = torch.Generator()
    random_number_generator_state_manager.manual_seed(torch.initial_seed() % 2**32)

    # Creates the data loader, which manages the loading of the dataset, it uses multiple worker processes, which load the dataset samples
    # asynchronously in the background (an worker initialization function and a generator are used to fix the seeds of the random number generators of
    # the worker process)
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=shuffle_samples,
        pin_memory=True,
        worker_init_fn=initialize_data_loader_worker,
        generator=random_number_generator_state_manager
    )


def initialize_data_loader_worker(_: int) -> None:
    """
    Creates an initialization function, which is used to fix the seeds of the Python built-in random number generator and the NumPy random number
    generator in the worker process that are used by the data loader to asynchronously load the dataset samples in the background

    Args:
        _ (int): Argument is not processed. Only signature is needed.
    """

    # Retrieves initial seed that was set during the application startup (this seed was either specified by the user as a command line argument,
    # or randomly generated, the seed that is used is the remainder of dividing the random seed by 2^32, which is done to ensure that the random
    # seed is a 32-bit unsigned integer, as NumPy requires the random seed to be a 32-bit unsigned integer)
    worker_random_seed = torch.initial_seed() % 2**32

    # Sets the random seeds of the built-in Python random number generator and the NumPy random number generator, the seed for the PyTorch random
    # number generator was already set using a PyTorch generator, which manages the state of the pseudo random number generator
    random.seed(worker_random_seed)
    numpy.random.seed(worker_random_seed)
