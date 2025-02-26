import os
import numpy as np
from PIL import Image

class CustomImageDataset:
    def __init__(self, image_dir, label_file=None, transform=None, batch_size=32, max_images_per_class=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            label_file (str, optional): Path to a label file (if labels are separate).
            transform (callable, optional): Optional transform to be applied on the image.
            batch_size (int): Maximum number of images per batch (optional).
            max_images_per_class (int, optional): Maximum number of images per class (optional).
        """
        self.image_dir = os.path.expanduser(image_dir)
        self.label_file = label_file
        self.transform = transform
        self.batch_size = batch_size
        self.max_images_per_class = max_images_per_class

        # Get all images and their corresponding labels (either from label file or directory structure)
        self.image_paths, self.labels, self.int_labels = self._load_images_and_labels(image_dir)

        self.total_images = len(self.image_paths)

        # Calculate optimal batch size, it should be the min between total_images and the provided batch_size
        self.optimal_batch_size = min(self.total_images, self.batch_size)

        # Calculate total number of batches
        self.total_batches = len(self.image_paths) // self.batch_size
        if len(self.image_paths) % self.batch_size != 0:
            self.total_batches += 1  # Add an additional batch for the remaining images

    def _load_images_and_labels(self, image_dir):
        """
        Load all image paths and their corresponding labels. If no label_file is provided,
        auto-labels based on the subdirectory structure.
        """
        image_paths = []
        labels = []
        int_labels = {}

        if self.label_file:
            # If label file is provided, load labels (this assumes label_file is a CSV with filenames and labels)
            with open(self.label_file, 'r') as f:
                for line in f:
                    img_name, label = line.strip().split(',')
                    img_path = os.path.join(image_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(int(label))
        else:
            # Auto-label based on subdirectory names
            class_names = sorted(os.listdir(image_dir))  # Get all subdirectories (class names)
            class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
            int_labels = class_to_idx

            # Loop through subdirectories (classes)
            for class_name, class_idx in class_to_idx.items():
                class_dir = os.path.join(image_dir, class_name)
                if os.path.isdir(class_dir):
                    class_images = []
                    # Collect images for the current class
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        if img_path.lower().endswith(('png', 'jpg', 'jpeg')):
                            class_images.append(img_path)

                    # If max_images_per_class is set, limit the number of images for this class
                    if self.max_images_per_class:
                        class_images = class_images[:self.max_images_per_class]

                    # Add the images from the current class to the overall list
                    for img_path in class_images:
                        image_paths.append(img_path)
                        labels.append(class_idx)

        return image_paths, labels, int_labels

    def __len__(self):
        # Return the total number of images
        return self.total_images

    def __getitem__(self, idx):
        """
        Get an image and its label based on index.
        """
        img_path = self.image_paths[idx]

        label = self.labels[idx]  # The label is already assigned in the self.labels list

        # Apply transformation if specified
        if self.transform:
            img = self.transform(img_path)
        else:
            img = Image.open(img_path).resize((224, 224)).convert('RGB')
            img = np.array(img)

        return img, label

    def __iter__(self):
        """
        Make the dataset iterable, yielding batches of images and labels.
        """
        batch = []
        for idx in range(self.total_images):
            img, label = self[idx]
            batch.append((img, label))

            # Yield batches of size optimal_batch_size
            if len(batch) == self.optimal_batch_size:
                images, labels = zip(*batch)  # Unzip the list of tuples into two lists
                yield np.stack(images), labels#np.array(labels)
                batch = []  # Reset batch

        # Yield remaining images if batch is not exactly divisible by optimal_batch_size
        if batch:
            images, labels = zip(*batch)
            yield np.stack(images), labels#np.array(labels)
