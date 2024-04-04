import sys
import os
parent_directory = os.getcwd()
sys.path.insert(0, parent_directory)
# print(sys.path)

import json


from adamatch.data import get_dataloaders
from adamatch.network import Encoder, Classifier
from adamatch.hyperparameters import adamatch_hyperparams
from adamatch.adamatch import Adamatch


if __name__ == '__main__':
    #* get source and target data
    data = get_dataloaders("./data", batch_size_source=10, workers=2)

    source_dataloader_train_weak, source_dataloader_train_strong = data[0]
    target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test = data[1]

    print(len(source_dataloader_train_weak))
    print(len(source_dataloader_train_strong))
    print(len(target_dataloader_train_weak))
    print(len(target_dataloader_train_strong))
    print(len(target_dataloader_test))


    #* instantiate the network
    encoder = Encoder()
    classifier = Classifier(n_classes=1)


    #* instantiate AdaMatch algorithm and setup hyperparameters
    adamatch = Adamatch(encoder, classifier)
    hparams = adamatch_hyperparams()
    epochs = 20
    save_path = "./results/adamatch_checkpoint.pt"

    #* train the model
    _, _, history = adamatch.train(
        source_dataloader_train_weak, source_dataloader_train_strong,
        target_dataloader_train_weak, target_dataloader_train_strong, target_dataloader_test,
        epochs, hparams, save_path
    )

    with open("./results/history.json", "w") as file:
        json.dump(history, file)