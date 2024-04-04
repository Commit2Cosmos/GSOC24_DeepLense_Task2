import sys
import os
parent_directory = os.getcwd()
sys.path.insert(0, parent_directory)

import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import sklearn
import seaborn as sns


from adamatch.adamatch import Adamatch
from adamatch.network import Encoder, Classifier
from adamatch.data import get_dataloaders



def plot_metrics(epoch_loss, accuracy_source):

    fig, axs = plt.subplots(1, 1, figsize=(18,5), dpi=200)

    epochs = len(epoch_loss)

    axs.plot(range(1, epochs+1), epoch_loss, label='loss')
    axs.plot(range(1, epochs+1), accuracy_source, label='source acc')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Acc/Loss')
    axs.set_title('Accuracies & Losses')
        
    plt.legend()
    plt.show()



def plot_cm_roc(eval_output, n_classes=2):
    """
    Plots the confusion matrix and ROC curves of the model on `dataloader`.

    Arguments:
    ----------
    dataloader: PyTorch DataLoader
        DataLoader with test data.

    n_classes: int
        Number of classes.
    """

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    _, labels_list, outputs_list, preds_list = eval_output

    #! plot confusion matrix
    cm = sklearn.metrics.confusion_matrix(labels_list, preds_list)
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['({0:.2%})'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(n_classes, n_classes)

    plt.figure(figsize=(8,4), dpi=200)
    print(cm.shape)
    print(labels.shape)
    sns.heatmap(cm, annot=labels, cmap=cmap, fmt="")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()

    #! plot roc
    #tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels_list, outputs_list)
    roc_auc = sklearn.metrics.auc(fpr, tpr)


    plt.figure(figsize=(7,5), dpi=200)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.show()



if __name__ == '__main__':
    encoder = Encoder()
    classifier = Classifier(n_classes=1)

    source_dataloader, test_dataloader = get_dataloaders("./data", batch_size_source=32, workers=2, train=False)

    encoder.eval()
    classifier.eval()
    ada = Adamatch(encoder, classifier, pretrained=True)


    print("Source accuracy:", ada.evaluate(source_dataloader))


    #! Loss & acc plotting
    # with open("./results/history.json", "r") as file:
    #     history = json.load(file)

    #* testing on fake data
    # size = 30
    # history = {}
    # history['epoch_loss'] = np.random.rand(size,)
    # history['accuracy_source'] = np.random.rand(size,)

    # plot_metrics(history['epoch_loss'], history['accuracy_source'])


    #! ROC plotting
    eval_data = ada.evaluate(test_dataloader, return_lists_roc=True)
    print("Test accuracy:", eval_data[0])
    plot_cm_roc(eval_data)