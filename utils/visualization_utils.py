from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_confusion_matrix(plot_title='Confusion Matrix', y_test=None, y_out=None):
    """
    Displays confusion matrix for classification problem

    :param plot_title: title to be added to plot
    :param y_test: target output data
    :param y_out: actual model output data
    :return: none
    """

    assert y_test is not None and y_out is not None, 'Error: test and output data must not by None'

    # Get and reshape confusion matrix data
    matrix = confusion_matrix(y_test, y_out)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=(16, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                cmap=plt.cm.Greens, linewidths=0.2)

    # Add labels to the plot
    class_names = [val for val in np.unique(y_out)]
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(plot_title)
    plt.show()
    plt.close()

