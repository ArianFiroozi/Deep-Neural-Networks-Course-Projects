import matplotlib.pyplot as plt
import numpy as np

def show_random_images(class_names, images, labels, predictions=None, num_images=5, label_flag=0):
    random_indices = np.random.choice(images.shape[0], num_images, replace=False)
    random_images = images[random_indices]
    random_labels = labels[random_indices]
    grid_size = int(np.ceil(np.sqrt(num_images)))
    plt.figure(figsize=(12, 12))
    if label_flag:
        random_preds = predictions[random_indices]
        for i, (image, label, pred) in enumerate(zip(random_images, random_labels, random_preds)):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(image.astype("uint8"))
            plt.title(f"True Label: {class_names[label]}\nPredicted Label: {class_names[pred]}", fontdict={'family': 'cursive'})
            plt.axis("off")
    else:
        for i, (image, label) in enumerate(zip(random_images, random_labels)):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(image.astype("uint8"))
            plt.title(f"True Label: {class_names[label]}", fontdict={'family': 'cursive'})
            plt.axis("off")
    plt.show()