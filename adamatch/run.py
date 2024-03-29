import sys
import os
parent_directory = os.getcwd()
sys.path.insert(0, parent_directory)
# print(sys.path)


from adamatch.data import get_dataloaders
from adamatch.network import Encoder, Classifier
from adamatch.hyperparameters import adamatch_hyperparams
from adamatch.adamatch import Adamatch


if __name__ == '__main__':
    #* get source and target data
    data = get_dataloaders("./data", batch_size_source=5, workers=2)

    source_dataloader_train_weak, source_dataloader_train_strong = data[0]
    target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

    print(len(source_dataloader_train_weak))
    print(len(source_dataloader_train_strong))
    print(len(target_dataloader_train_weak))
    print(len(target_dataloader_train_strong))
    print(len(target_dataloader_test))


    #* instantiate the network
    n_classes = 2
    encoder = Encoder()
    classifier = Classifier(n_classes=n_classes)


    #* instantiate AdaMatch algorithm and setup hyperparameters
    adamatch = Adamatch(encoder, classifier)
    hparams = adamatch_hyperparams()
    epochs = 2
    save_path = "./adamatch_checkpoint.pt"

    #* train the model
    adamatch.train(source_dataloader_train_weak, source_dataloader_train_strong,
                target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
                epochs, hparams, save_path)

    #* evaluate the model
    # adamatch.plot_metrics()

    #* returns accuracy on the test set
    # print(f"accuracy on test set = {adamatch.evaluate(target_dataloader_test)}")

    #* returns a confusion matrix plot and a ROC curve plot (that also shows the AUROC)
    # adamatch.plot_cm_roc(target_dataloader_test)