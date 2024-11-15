import numpy as np

def image_augmentation(images, labels, augmentations, num_augmentations=1):
    images = images.astype(np.uint8)
    augmented_images = []
    for image in images:
        for _ in range(num_augmentations):
            augmented = augmentations(image=image)
            augmented_images.append(augmented['image'])
    return np.array(augmented_images), np.repeat(labels, num_augmentations)