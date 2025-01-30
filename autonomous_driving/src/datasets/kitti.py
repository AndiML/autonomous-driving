import os
import zipfile
from matplotlib import pyplot as plt
import requests
import shutil
import yaml

from glob import glob
from PIL import Image
from tqdm import tqdm


import numpy
import torch
import cv2

from autonomous_driving.src.datasets.dataset import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2


class KITTICustom(Dataset):
    """
    Custom Dataset class for preparing the KITTI dataset in YOLOv8-compatible format.
    """
    URL_IMAGE_DATA = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    URL_LABEL_ANNOTATIONS = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

    CLASS_TO_INDEX = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Truck": 3,
    "Van": 4,
    "Tram": 5,
    "Person_sitting": 6
    }

    def __init__(self, root: str, split: str ='train', transform=None, download: bool=False, split_ratio: float=0.8):
        """
        Initializes the KITTI Custom Dataset.

        Args:
            root (str): Root directory where the dataset will be stored.
            split (str): One of 'train', 'val', or 'test'.
            transform (albumentations.Compose): Albumentations transformations to apply.
            download (bool): If True, downloads and preprocesses the dataset.
            split_ratio (float): Ratio of data to be used for training.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.split_ratio = split_ratio

        # Define directories
        self.dataset_dir = os.path.join(root, "kitti_dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.labels_dir = os.path.join(self.dataset_dir, "labels")
        self.train_images_dir = os.path.join(self.images_dir, "train")
        self.val_images_dir = os.path.join(self.images_dir, "val")
        self.test_images_dir = os.path.join(self.images_dir, "test")
        self.train_labels_dir = os.path.join(self.labels_dir, "train")
        self.val_labels_dir = os.path.join(self.labels_dir, "val")


        if download:
            self._download()
            self._preprocess_labels()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        self.images, self.labels = self._load_data()

    def _download(self):
        """Download and extract the KITTI dataset."""
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)

            # Download and extract image data
            print("Downloading KITTI image data...")
            image_zip_path = os.path.join(self.root, "kitti_images.zip")
            self._download_file(self.URL_IMAGE_DATA, image_zip_path)
            print("Extracting KITTI image data...")
            self._extract_zip(image_zip_path, self.dataset_dir)
            os.remove(image_zip_path)

            # Download and extract label annotations
            print("Downloading KITTI label annotations...")
            label_zip_path = os.path.join(self.root, "kitti_labels.zip")
            self._download_file(self.URL_LABEL_ANNOTATIONS, label_zip_path)
            print("Extracting KITTI label annotations...")
            self._extract_zip(label_zip_path, self.dataset_dir)
            os.remove(label_zip_path)

            print("Download and extraction complete.")
        else:
            print("KITTI dataset already exists. Skipping download.")

    def _download_file(self, url, path):
        """Download a file from a URL with a progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(path, 'wb') as file, tqdm(
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            desc=f"Downloading {os.path.basename(path)}"
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

    def _extract_zip(self, zip_path, extract_to):
        """Extract a zip file with a progress bar."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_infos = zip_ref.infolist()
            with tqdm(total=len(zip_infos), desc=f"Extracting {os.path.basename(zip_path)}") as bar:
                for zip_info in zip_infos:
                    zip_ref.extract(zip_info, extract_to)
                    bar.update(1)

    def _preprocess_labels(self)-> None:
        """
        Convert KITTI labels to YOLO format and organize dataset into train and val directories.
        """
        print("Preprocessing labels and organizing dataset structure...")

        # Paths to original images and labels
        original_train_images_dir = os.path.join(self.dataset_dir, "training", "image_2")
        original_train_labels_dir = os.path.join(self.dataset_dir, "training", "label_2")
        original_test_images_dir = os.path.join(self.dataset_dir, "testing", "image_2")
        original_images = glob(os.path.join(original_train_images_dir, "*.png"))

        # Shuffle and split
        shuffled_indices = numpy.random.permutation(len(original_images))
        train_size = int(self.split_ratio * len(original_images))
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]

        splits = {
            'train': train_indices,
            'val': val_indices
        }
        # Genrate train and validation data
        for split, indices in splits.items():
            images_dest_dir = self.train_images_dir if split == 'train' else self.val_images_dir
            labels_dest_dir = self.train_labels_dir if split == 'train' else self.val_labels_dir

            os.makedirs(images_dest_dir, exist_ok=True)
            os.makedirs(labels_dest_dir, exist_ok=True)

            for index in tqdm(indices, desc=f"Processing {split.upper()} split"):
                img_path = original_images[index]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(original_train_labels_dir, f"{img_name}.txt")

                # Define destination paths
                dest_img_path = os.path.join(images_dest_dir, os.path.basename(img_path))
                dest_label_path = os.path.join(labels_dest_dir, f"{img_name}.txt")
                # Copy image
                if not os.path.exists(dest_img_path):
                    shutil.copyfile(img_path, dest_img_path)

                # Convert and copy label
                if os.path.exists(label_path):
                    self._convert_label_to_yolo(label_path, dest_label_path, dest_img_path)
                else:
                    # If no label exists, create an empty label file
                    open(dest_label_path, 'w').close()


        # Process test split
        original_test_images = glob(os.path.join(original_test_images_dir, "*.png"))
        os.makedirs(self.test_images_dir, exist_ok=True)
        for img_path in tqdm(original_test_images, desc="Copying test images"):
            img_name = os.path.basename(img_path)
            dest_img_path = os.path.join(self.test_images_dir, img_name)
            if not os.path.exists(dest_img_path):
                shutil.copyfile(img_path, dest_img_path)

        shutil.rmtree(os.path.join(self.dataset_dir, "training"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.dataset_dir, "testing"), ignore_errors=True)

        print("Preprocessing complete.")

    def _convert_label_to_yolo(self, src_label_path: str, dest_label_path:str , image_path: str) -> None:
        """Convert KITTI label format to YOLO format."""
        with open(src_label_path, 'r') as f:
            lines = f.readlines()
        image = Image.open(image_path)
        width, height = image.size

        yolo_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8:
                continue  # Incomplete label
            class_name = parts[0]
            if class_name not in self.CLASS_TO_INDEX:
                continue  # Ignore classes not in the specified list
            class_id = self.CLASS_TO_INDEX[class_name]
            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Clamp values to [0,1]
            x_center = numpy.clip(x_center, 0, 1)
            y_center = numpy.clip(y_center, 0, 1)
            w = numpy.clip(w, 0, 1)
            h = numpy.clip(h, 0, 1)

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write to destination label file
        with open(dest_label_path, 'w') as f:
            f.write("\n".join(yolo_labels))


    def _load_data(self):
        """Load image paths and corresponding label paths based on the split."""
        if self.split == 'test':
            images_dir = self.test_images_dir
            labels = []
            images = sorted(glob(os.path.join(images_dir, "*.png")))
        else:
            images_dir = self.train_images_dir if self.split == 'train' else self.val_images_dir
            labels_dir = self.train_labels_dir if self.split == 'train' else self.val_labels_dir

            images = sorted(glob(os.path.join(images_dir, "*.png")))
            label_files = sorted(glob(os.path.join(labels_dir, "*.txt")))
            labels = [self._parse_label_file(label_file) for label_file in label_files]

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        img_path = self.images[index]
        label = self.labels[index] if self.split != 'test' else None

        image = numpy.array(Image.open(img_path))

        bboxes, category_ids = label if label else ([], [])

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']

        if self.split != 'test':
            labels = {
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'labels': torch.tensor(category_ids, dtype=torch.long)
            }
            return image, labels
        else:
            return image

    def _parse_label_file(self, label_path):
        with open(label_path, 'r') as f:
            label_data = f.read()
        return self._parse_label(label_data)

    def _parse_label(self, label_data):
        """Parse YOLO-formatted label data."""
        bboxes = []
        category_ids = []
        for line in label_data.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Invalid label format
            class_id, x_center, y_center, w, h = parts
            bboxes.append([float(x_center), float(y_center), float(w), float(h)])
            category_ids.append(int(class_id))
        return bboxes, category_ids

    def _check_exists(self):
        """Check if the dataset directories exist."""
        required_dirs = [
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir,
            self.test_images_dir
        ]
        return all(os.path.exists(d) for d in required_dirs)



    def _generate_yaml(self):
        """Generate the YAML configuration file for YOLO model."""

        yaml_path = os.path.join(self.dataset_dir, "kitti.yaml")

        data = {
            'train': os.path.join('kitti_dataset', 'images', 'train'),
            'val': os.path.join('kitti_dataset', 'images', 'val'),
            'test': os.path.join('kitti_dataset', 'images', 'test'),
            'nc': len(self.CLASS_TO_INDEX),
            'names': list(self.CLASS_TO_INDEX.values())
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        print(f"YOLO YAML configuration saved to {yaml_path}")


    def get_labels(self) -> list[str]:
        """Retrieve the list of class names."""
        return list(self.CLASS_TO_INDEX.keys())

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self

    @property
    def number_of_classes(self) -> int:
        return self

class KITTI(Dataset):
    """
    KITTI Dataset class tailored for YOLO.
    """
    dataset_id = 'kitti'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str, split_ratio:float=0.8, customize_dataset: bool=False) -> None:
        """
        Initializes the KITTI dataset.

        Args:
            path (str): Path where the KITTI dataset is stored or will be downloaded to.
            split_ratio (float): Ratio of data to be used for training.
            customize_dataset (bool): If True, allows users to apply custom data augmentations and handle the dataset loading process independently,
            rather than relying on the default pipeline.
        """
        self.path = path
        self.split_ratio = split_ratio
        self.name = 'KITTI'
        if not customize_dataset:
        # Define augmentation transforms using albumentations
            self.train_transform = albumentations.Compose([
                albumentations.Resize(height=640, width=640),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.1),
                albumentations.Rotate(limit=15, p=0.5),
                albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                albumentations.GaussianBlur(blur_limit=3, p=0.3),
                albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']))


            self.val_transform = albumentations.Compose([
                albumentations.Resize(height=640, width=640),
                ToTensorV2()
            ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']))

            self.test_transform = albumentations.Compose([
                albumentations.Resize(height=640, width=640),
                ToTensorV2()
            ])  # No bbox_params for test as there are no labels

            # Initialize training and validation datasets
            self._training_data = KITTICustom(
                root=self.path,
                split='train',
                download=True,
                transform=self.train_transform,
                split_ratio=split_ratio
            )
            self._validation_data = KITTICustom(
                root=self.path,
                split='val',
                download=False,
                transform=self.val_transform,
                split_ratio=split_ratio
            )
            self._test_data = KITTICustom(
                root=self.path,
                split='test',
                download=False,
                transform=self.test_transform
            )


    def get_labels(self) -> list[int]:
        """Retrieve the list of class names."""
        return list(self.training_data.CLASS_TO_INDEX.keys())

    @property
    def training_data(self) -> KITTICustom:
        """Get the training dataset."""
        return self._training_data

    @property
    def validation_data(self) -> KITTICustom:
        """Get the validation dataset."""
        return self._validation_data

    @property
    def test_data(self) -> KITTICustom:
        """Get the test dataset."""
        return self._test_data

    @property
    def number_of_classes(self) -> int:
        """Number of distinct classes."""
        return len(self.class_names)

    @property
    def sample_shape(self) -> tuple:
        """Returns the shape of the input samples."""
        return self.training_data[0].shape


    @staticmethod
    def download(self) -> None:
        """Download  the KITTI dataset and generate  YAML configuration file"""
        KITTICustom(root=self.path, download=True, transform=None, split_ratio=self.split_ratio)

        # Generate YAML configuration
        KITTICustom._generate_yaml()



