import matplotlib.pyplot as plt
import numpy as np

def show_random_images(images, masks, predictions=None, num_samples=5, label_flag=0):
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(12, 4*num_samples))
    if label_flag:
        for i, idx in enumerate(indices):
            plt.subplot(num_samples, 3, 3 * i + 1)
            plt.imshow(images[idx], cmap=None)
            plt.title(f"Image {idx}")
            plt.subplot(num_samples, 3, 3 * i + 2)
            plt.imshow(masks[idx], cmap='gray')
            plt.title(f"Mask {idx}")
            plt.subplot(num_samples, 3, 3 * i + 3)
            plt.imshow(predictions[idx], cmap='gray')
            plt.title(f"Pred {idx}")
    else:
        for i, idx in enumerate(indices):
            plt.subplot(num_samples, 2, 2 * i + 1)
            plt.imshow(images[idx], cmap=None)
            plt.title(f"Image {idx}")
            plt.subplot(num_samples, 2, 2 * i + 2)
            plt.imshow(masks[idx], cmap='gray')
            plt.title(f"Mask {idx}")
    plt.tight_layout()
    plt.show()