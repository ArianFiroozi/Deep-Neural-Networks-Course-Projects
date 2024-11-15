import matplotlib.pyplot as plt
import numpy as np

def show_random_images(images, labels, class_names, num_images=9):
    random_indices = np.random.choice(images.shape[0], num_images, replace=False)
    random_images = images[random_indices]
    random_labels = labels[random_indices]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(zip(random_images, random_labels)):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.title(class_names[label], fontdict={'family': 'cursive'})
        plt.axis("off")
    plt.show()