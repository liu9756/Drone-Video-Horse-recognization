from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils import get_confusion_matrix


def plot_confusion_matrix(cm,index_to_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d") 
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(ticks=np.arange(len(index_to_name))+0.5, labels=index_to_name, rotation=45)
    plt.yticks(ticks=np.arange(len(index_to_name))+0.5, labels=index_to_name, rotation=45)
    plt.tight_layout()
    plt.show()




