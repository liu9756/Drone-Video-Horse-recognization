import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, save_path='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    cm = np.array([
        [49363, 4251, 1845, 1195, 592, 370, 233],
        [13164, 34933, 1331, 4895, 1736, 116, 2108],
        [2883, 97, 42971, 53, 7800, 22, 80],
        [4187, 2515, 188, 41942, 461, 11, 56],
        [886, 1557, 7545, 631, 46964, 0, 440],
        [861, 1017, 346, 3009, 205, 49234, 100],
        [675, 684, 475, 271, 3244, 24, 52910]
    ])
    classes = ['Horse_1_crop', 'Horse_2_crop', 'Horse_3_crop', 'Horse_4_crop', 'Horse_5_crop', 'Horse_6_crop', 'Horse_7_crop']
    plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path='confusion_matrix.png')
