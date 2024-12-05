import os
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split

def load_tiff_image(image_path):
    return tiff.imread(image_path)

def dataset_loader(dataset_dir):
    image_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".tif") and not file.endswith("_mask.tif"):
                image_files.append(os.path.join(root, file))
    train_images, test_images = train_test_split(image_files, test_size=0.2, random_state=24)
    val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=24)
 
    def load_files(image_paths):
        images = []
        masks = []
        for image_path in image_paths:
            image_name = os.path.basename(image_path)            
            image_array = load_tiff_image(image_path)
            mask_name = image_name.replace('.tif', '_mask.tif')
            mask_path = os.path.join(os.path.dirname(image_path), mask_name)
            if os.path.exists(mask_path):
                mask_array = load_tiff_image(mask_path)
                images.append(image_array)
                masks.append(mask_array)
        return np.array(images), np.array(masks)
    train_images, train_masks = load_files(train_images)
    val_images, val_masks = load_files(val_images)
    test_images, test_masks = load_files(test_images) 
    print("Dataset loaded into numpy arrays successfully!")
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)