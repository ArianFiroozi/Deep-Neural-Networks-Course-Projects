import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy_loss_plot(hist):
    plt.subplot(2, 2, 1)
    plt.plot(hist.history['accuracy'], label='train')
    plt.plot(hist.history['val_accuracy'], label='validation')
    plt.title('accuracy over epoches', fontdict={'family': 'cursive'})
    plt.legend(loc='lower right')
    plt.subplot(2, 2, 2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('loss over epoches', fontdict={'family': 'cursive'})
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 3)
    plt.plot(hist.history['iou_metric'], label='train')
    plt.plot(hist.history['val_iou_metric'], label='validation')
    plt.title('IoU over epoches', fontdict={'family': 'cursive'})
    plt.legend(loc='lower right')
    plt.subplot(2, 2, 4)
    plt.plot(hist.history['dice_coefficient'], label='train')
    plt.plot(hist.history['val_dice_coefficient'], label='validation')
    plt.title('dice coefficient over epoches', fontdict={'family': 'cursive'})
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()