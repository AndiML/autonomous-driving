import os
import zipfile
import requests
from glob import glob
from pathlib import Path
from PIL import Image
import shutil

import numpy as np
import torch

from tqdm import tqdm
from autonomous_driving.src.datasets.dataset import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2

import yaml

class KITTICustom(Dataset):
    """
    Custom Dataset class for preparing the KITTI dataset in YOLOv8-compatible format.
    """
    URL_IMAGE_DATA = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    URL_LABEL_ANNOTATIONS = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

    LABEL_TO_INDEX = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    def __init__(self, root, train=True, transform=None, download=False, split_ratio=0.8):
        """
        Initializes the KITTI Custom Dataset.

        Args:
            root (str): Root directory where the dataset will be stored.
            train (bool): If True, prepares the training set; otherwise, the validation set.
            transform (albumentations.Compose): Albumentations transformations to apply.
            download (bool): If True, downloads and preprocesses the dataset.
            split_ratio (float): Ratio of data to be used for training.
        """
        self.root = root
        self.train = train
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
            # self._download()
            self._preprocess_labels()
        exit()
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

    def _preprocess_labels(self):
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
        shuffled_indices = np.random.permutation(len(original_images))
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
                img_path = Path(original_images[index])
                img_name = img_path.stem
                label_path = os.path.join(original_train_labels_dir, f"{img_name}.txt")

                # Define destination paths
                dest_img_path = os.path.join(images_dest_dir, img_path.name)
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
        print("Processing test data...")
        original_test_images = glob(os.path.join(original_test_images_dir, "*.png"))
        os.makedirs(self.test_images_dir, exist_ok=True)

        for img_path in tqdm(original_test_images, desc="Copying test images"):
            img_name = Path(img_path).name
            dest_img_path = os.path.join(self.test_images_dir, img_name)
            if not os.path.exists(dest_img_path):
                shutil.copyfile(img_path, dest_img_path)

        print("Preprocessing complete.")

    def _convert_label_to_yolo(self, src_label_path, dest_label_path, image_path):
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
            if class_name not in self.LABEL_TO_INDEX:
                continue  # Ignore classes not in the specified list
            class_id = self.LABEL_TO_INDEX[class_name]
            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])

            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            # Clamp values to [0,1]
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Write to destination label file
        with open(dest_label_path, 'w') as f:
            f.write("\n".join(yolo_labels))

    def _load_data(self):
        """Load image paths and corresponding label paths."""
        images_dir = self.train_images_dir if self.train else self.val_images_dir
        labels_dir = self.train_labels_dir if self.train else self.val_labels_dir

        images = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.png')])
        labels = sorted([os.path.join(labels_dir, lbl) for lbl in os.listdir(labels_dir) if lbl.endswith('.txt')])

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Retrieve the image and label at the specified index.

        Args:
            index (int): Index of the data point.

        Returns:
            image (torch.Tensor): Transformed image tensor.
            label (dict): Dictionary containing bounding boxes and labels.
        """
        img_path = self.images[index]
        label_path = self.labels[index]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Initialize bboxes and category_ids
        bboxes = []
        category_ids = []

        # Load label
        with open(label_path, 'r') as f:
            label_data = f.read()
            if label_data.strip():  # Ensure the label file is not empty
                bboxes, category_ids = self._parse_label(label_data)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed['image']
            bboxes = transformed.get('bboxes', [])
            category_ids = transformed.get('category_ids', [])

        # Convert to tensors
        if bboxes:
            labels = {
                'boxes': torch.tensor(bboxes, dtype=torch.float32),
                'labels': torch.tensor(category_ids, dtype=torch.long)
            }
        else:
            labels = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.long)
            }

        return image, labels

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

    def get_labels(self) -> dict[str, int]:
        """Get the label mapping."""
        return self.LABEL_TO_INDEX

    @property
    def number_of_classes(self) -> int:
        """Returns the number of distinct classes."""
        return len(self.LABEL_TO_INDEX)

    @property
    def sample_shape(self) -> tuple:
        """Returns the shape of the input samples."""
        return (3, 480, 640)  # Example: (channels, height, width)

class KITTI(Dataset):
    """
    KITTI Dataset class tailored for YOLOv8.
    """
    dataset_id = 'kitti'
    """Contains a machine-readable ID that uniquely identifies the dataset."""

    def __init__(self, path: str, split_ratio=0.8):
        """
        Initializes the KITTI dataset.

        Args:
            path (str): Path where the KITTI dataset is stored or will be downloaded to.
            split_ratio (float): Ratio of data to be used for training.
        """
        self.path = path
        self.name = 'KITTI'

        # Define augmentation transforms using albumentations
        self.train_transform = albumentations.Compose([
            albumentations.Resize(640, 480),
            albumentations.RandomCrop(640, 480),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Rotate(limit=15, p=0.5),
            albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']))

        self.val_transform = albumentations.Compose([
            albumentations.Resize(640, 480),
            ToTensorV2()
        ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['category_ids']))

        # Initialize training and validation datasets
        self._training_data = KITTICustom(root=self.path, train=True, download=True, transform=self.train_transform, split_ratio=split_ratio)
        self._validation_data = KITTICustom(root=self.path, train=False, download=False, transform=self.val_transform, split_ratio=split_ratio)

        # Extract class names
        self.class_names = list(self._training_data.get_labels().keys())

        # Generate YAML configuration
        self.yaml_path = os.path.join(self.path, "kitti.yaml")
        self._generate_yaml()

    def _generate_yaml(self):
        """Generate the YAML configuration file for YOLOv8."""
        data = {
            'train': os.path.join(self.path, 'kitti_dataset', 'images', 'train'),
            'val': os.path.join(self.path, 'kitti_dataset', 'images', 'val'),
            'nc': len(self.class_names),
            'names': self.class_names
        }

        with open(self.yaml_path, 'w') as f:
            yaml.dump(data, f)

        print(f"YOLOv8 YAML configuration saved to {self.yaml_path}")

    def get_labels(self) -> list:
        """Retrieve the list of class names."""
        return self.class_names

    @property
    def training_data(self) -> KITTICustom:
        """Get the training dataset."""
        return self._training_data

    @property
    def validation_data(self) -> KITTICustom:
        """Get the validation dataset."""
        return self._validation_data

    @property
    def number_of_classes(self) -> int:
        """Number of distinct classes."""
        return len(self.class_names)

    @staticmethod
    def download(path: str, split_ratio=0.8) -> None:
        """Download and prepare the KITTI dataset.

        Args:
            path (str): Directory where the dataset will be stored.
            split_ratio (float): Ratio of data to be used for training.
        """
        KITTICustom(root=path, train=True, download=True, transform=None, split_ratio=split_ratio)
        KITTICustom(root=path, train=False, download=False, transform=None, split_ratio=split_ratio)


# # Example Usage
# if __name__ == "__main__":
#     # Define the root directory for the dataset
#     dataset_root = '/path/to/dataset/root'  # Replace with your desired path

#     # Initialize and prepare the KITTI dataset
#     kitti_dataset = KITTI(path=dataset_root)

#     # Verify the YAML file
#     yaml_path = kitti_dataset.yaml_path
#     print(f"YAML configuration located at: {yaml_path}")

#     # Initialize YOLOv8 model (Ensure you have the Ultralytics package installed)
#     # Install with: pip install ultralytics
#     from ultralytics import YOLO

#     # Initialize the model (e.g., YOLOv8s)
#     model = YOLO('yolov8s.pt')  # You can choose other variants like 'yolov8m.pt', 'yolov8l.pt', etc.

#     # Train the model
#     model.train(data=yaml_path, epochs=100, imgsz=640)

#     # Evaluate the model
#     results = model.val()

#     # Inference example
#     # Replace '/path/to/image.jpg' with an actual image path
#     # results = model('/path/to/image.jpg')
#     # results.show()
