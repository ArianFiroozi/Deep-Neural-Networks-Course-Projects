import numpy as np

def image_augmentation(images, masks, augmentations, num_augmentations=1):
    images = images.astype(np.uint8)
    masks = masks.astype(np.uint8)
    augmented_images = []
    augmented_masks = []
    for image, mask in zip(images, masks):
        for _ in range(num_augmentations):
            augmented = augmentations(image=image, mask=mask)
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])
    return np.array(augmented_images), np.array(augmented_masks)