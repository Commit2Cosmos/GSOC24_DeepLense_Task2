import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import sklearn.metrics

class Adamatch():
    """
    Paper: AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation
    Authors: David Berthelot, Rebecca Roelofs, Kihyuk Sohn, Nicholas Carlini, Alex Kurakin
    """

    def __init__(self, encoder, classifier, pretrained = False):
        """
        NOTE: the actual AdaMatch paper doesn't separate between encoder and classifier,
        but I find it more practical for the purposes of setting up the networks.

        Arguments:
        ----------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        
        if pretrained:
            weights = torch.load("./results/adamatch_checkpoint.pt", map_location=torch.device('cpu'))
            self.encoder.load_state_dict(weights["encoder_weights"])
            self.classifier.load_state_dict(weights["classifier_weights"])

    def train(self, source_dataloader_weak, source_dataloader_strong,
              target_dataloader_weak, target_dataloader_strong, target_dataloader_test,
              epochs, hyperparams, save_path):
        """
        Trains the model (encoder + classifier).

        Arguments:
        ----------
        source_dataloader_weak: PyTorch DataLoader
            DataLoader with source domain training data with weak augmentations.

        source_dataloader_strong: PyTorch DataLoader
            DataLoader with source domain training data with strong augmentations.

        target_dataloader_weak: PyTorch DataLoader
            DataLoader with target domain training data with weak augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE.

        target_dataloader_strong: PyTorch DataLoader
            DataLoader with target domain training data with strong augmentations.
            THIS DATALOADER'S BATCH SIZE MUST BE 3 * SOURCE_DATALOADER_BATCH_SIZE. 

        target_dataloader_test: PyTorch DataLoader
            DataLoader with target domain validation data, used for early stopping.

        epochs: int
            Amount of epochs to train the model for.

        hyperparams: dict
            Dictionary containing hyperparameters for this algorithm. Check `data/hyperparams.py`.

        save_path: str
            Path to store model weights.

        Returns:
        --------
        encoder: PyTorch neural network
            Neural network that receives images and encodes them into an array of size X.

        classifier: PyTorch neural network
            Neural network that receives an array of size X and classifies it into N classes.
        """

        # configure hyperparameters
        lr = hyperparams['learning_rate']
        wd = hyperparams['weight_decay']
        step_scheduler = hyperparams['step_scheduler']
        tau = hyperparams['tau']
        
        iters = max(len(source_dataloader_weak), len(source_dataloader_strong), len(target_dataloader_weak), len(target_dataloader_strong))

        # mu related stuff
        steps_per_epoch = iters
        total_steps = epochs * steps_per_epoch 
        current_step = 0

        # configure optimizer and scheduler
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.classifier.parameters()), lr=lr, weight_decay=wd)
        if step_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20)

        # early stopping variables
        start_epoch = 0
        best_acc = 0.0
        patience = 20
        bad_epochs = 0

        self.history = {'epoch_loss': [], 'accuracy_source': [], 'accuracy_target': []}

        #* training loop
        print("Training started")

        for epoch in range(start_epoch, epochs):
            running_loss = 0.0

            # set network to training mode
            self.encoder.train()
            self.classifier.train()

            dataset = zip(source_dataloader_weak, source_dataloader_strong, target_dataloader_weak, target_dataloader_strong)

            #* this is where the unsupervised learning comes in, as such, we're not interested in labels
            for (data_source_weak, labels_source), (data_source_strong, _), (data_target_weak, _), (data_target_strong, _) in dataset:
                data_source_weak = data_source_weak.to(self.device)
                labels_source = labels_source.to(self.device)

                data_source_strong = data_source_strong.to(self.device)
                data_target_weak = data_target_weak.to(self.device)
                data_target_strong = data_target_strong.to(self.device)

                #* concatenate data (in case of low GPU power this could be done after classifying with the model)
                data_combined = torch.cat([data_source_weak, data_source_strong, data_target_weak, data_target_strong], 0)
                source_combined = torch.cat([data_source_weak, data_source_strong], 0)

                #* get source data limit (useful for slicing later)
                source_total = source_combined.size(0)

                #* zero gradients
                optimizer.zero_grad()

                #* forward pass: calls the model once for both source and target and once for source only
                logits_combined = self.classifier(self.encoder(data_combined)).squeeze(1)
                logits_source_p = logits_combined[:source_total]
                # print("logits_source_p: ", logits_source_p)

                #* from https://github.com/yizhe-ang/AdaMatch-PyTorch/blob/main/trainers/adamatch.py
                self._disable_batchnorm_tracking(self.encoder)
                self._disable_batchnorm_tracking(self.classifier)
                logits_source_pp = self.classifier(self.encoder(source_combined)).squeeze(1)
                # print("logits_source_pp: ", logits_source_pp)
                
                self._enable_batchnorm_tracking(self.encoder)
                self._enable_batchnorm_tracking(self.classifier)

                #* perform random logit interpolation
                lambd = torch.rand_like(logits_source_p).to(self.device)
                final_logits_source = (lambd * logits_source_p) + ((1-lambd) * logits_source_pp)
                
                #* distribution allignment
                ## softmax for logits of weakly augmented source images
                logits_source_weak = final_logits_source[:data_source_weak.size(0)]
                pseudolabels_source = F.sigmoid(logits_source_weak)
                # print("pseudolabels_source: ", pseudolabels_source)

                #* softmax for logits of weakly augmented target images
                logits_target = logits_combined[source_total:]
                logits_target_weak = logits_target[:data_target_weak.size(0)]
                pseudolabels_target = F.sigmoid(logits_target_weak)
                # print("pseudolabels_target: ", pseudolabels_target)


                #* allign target label distribtion to source label distribution
                expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
                # print("expectation_ratio: ", expectation_ratio)

                #* L2 normalization
                def l2_norm(x):
                    return x / torch.sqrt(2 * x ** 2 - 2 * x + 1)
                
                final_pseudolabels = l2_norm(pseudolabels_target * expectation_ratio)
                # final_pseudolabels = pseudolabels_target * expectation_ratio
                # print("final_pseudolabels: ", final_pseudolabels)

                #* perform relative confidence thresholding
                max_binary = torch.where(pseudolabels_source < 0.5, 1-pseudolabels_source, pseudolabels_source)
                final_sum = torch.mean(max_binary, 0)
                
                #* define relative confidence threshold
                c_tau = tau * final_sum
                # print("c_tau: ", c_tau)


                mask = (final_pseudolabels >= c_tau).float()
                print("mask: ", torch.count_nonzero(mask).item())

                #* compute loss
                source_loss = self._compute_source_loss(logits_source_weak, final_logits_source[data_source_weak.size(0):], labels_source)
                # print("source loss: ", source_loss.item())
                
                final_pseudolabels = torch.round(final_pseudolabels)

                target_loss = self._compute_target_loss(final_pseudolabels, logits_target[data_target_weak.size(0):], mask)
                # print("target loss: ", target_loss)

                #* compute target loss weight (mu)
                pi = torch.tensor(np.pi, dtype=torch.float).to(self.device)
                step = torch.tensor(current_step, dtype=torch.float).to(self.device)
                mu = 0.5 - torch.cos(torch.minimum(pi, (2*pi*step) / total_steps)) / 2

                #* get total loss
                loss = source_loss + (mu * target_loss)
                current_step += 1

                #* backpropagate and update weights
                loss.backward()
                optimizer.step()

                # print("encoder.parameters: ", loss.grad)
                # print("classifier.parameters: ", self.classifier.parameters().grad)
                

                #* metrics
                running_loss += loss.item()

            #* get losses
            epoch_loss = running_loss / iters
            self.history['epoch_loss'].append(epoch_loss)

            #* evaluate on testing data (target domain)
            epoch_accuracy_source = self.evaluate(source_dataloader_weak)
            test_epoch_accuracy = self.evaluate(target_dataloader_test)
            
            self.history['accuracy_source'].append(epoch_accuracy_source)

            #* save checkpoint
            if test_epoch_accuracy > best_acc:
                torch.save({'encoder_weights': self.encoder.state_dict(),
                            'classifier_weights': self.classifier.state_dict()
                            }, save_path)
                best_acc = test_epoch_accuracy
                bad_epochs = 0
                
            else:
                bad_epochs += 1
                
            print('[Epoch {}/{}] loss: {:.6f}; accuracy source: {:.6f}; val accuracy: {:.6f};'.format(epoch+1, epochs, epoch_loss, epoch_accuracy_source, test_epoch_accuracy))
            # print('[Epoch {}/{}] loss: {:.6f}; accuracy source: {:.6f}'.format(epoch+1, epochs, epoch_loss, epoch_accuracy_source))
            
            if bad_epochs >= patience:
                print(f"Reached {bad_epochs} bad epochs, stopping training with best val accuracy of {best_acc}!")
                break

            #* scheduler step
            if step_scheduler:
                scheduler.step()

        best = torch.load(save_path)
        self.encoder.load_state_dict(best['encoder_weights'])
        self.classifier.load_state_dict(best['classifier_weights'])
        
        return self.encoder, self.classifier, self.history

    def evaluate(self, dataloader, return_lists_roc=False):
        """
        Evaluates model on `dataloader`.

        Arguments:
        ----------
        dataloader: PyTorch DataLoader
            DataLoader with test data.

        return_lists_roc: bool
            If True returns also list of labels, a list of outputs and a list of predictions.
            Useful for some metrics.

        Returns:
        --------
        accuracy: float
            Accuracy achieved over `dataloader`.
        """

        # set network to evaluation mode
        self.encoder.eval()
        self.classifier.eval()

        labels_list = []
        outputs_list = []
        preds_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # predict
                # print(self.classifier(self.encoder(data)))
                outputs = self.classifier(self.encoder(data)).cpu().squeeze(1)

                # numpify
                labels_numpy = labels.detach().cpu().numpy()
                outputs_numpy = outputs.detach().numpy() # probs (AUROC)

                preds = F.sigmoid(outputs).numpy() # accuracy
                

                # append
                labels_list.append(labels_numpy)
                outputs_list.append(outputs_numpy)
                preds_list.append(preds)

            labels_list = np.concatenate(labels_list)
            outputs_list = np.concatenate(outputs_list)
            preds_list = np.concatenate(preds_list)

        # metrics
        #auc = sklearn.metrics.roc_auc_score(labels_list, outputs_list, multi_class='ovr')
        labels_list = np.where(labels_list > 0.5, 1, 0)
        preds_list = np.where(preds_list > 0.5, 1, 0)
        accuracy = sklearn.metrics.accuracy_score(labels_list, preds_list)

        if return_lists_roc:
            return accuracy, labels_list, outputs_list, preds_list
            
        return accuracy


    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_source_loss(logits_weak, logits_strong, labels):
        """
        Receives logits as input (dense layer outputs with no activation function)
        """
        loss_function = nn.BCEWithLogitsLoss() # default: `reduction="mean"`

        weak_loss = loss_function(logits_weak, labels)
        strong_loss = loss_function(logits_strong, labels)

        #return weak_loss + strong_loss
        return (weak_loss + strong_loss) / 2

    @staticmethod
    def _compute_target_loss(pseudolabels, logits_strong, mask):
        """
        Receives logits as input (dense layer outputs with no activation function).
        `pseudolabels` are treated as ground truth (standard SSL practice).
        """
        loss_function = nn.BCEWithLogitsLoss(reduction="none")
        pseudolabels = pseudolabels.detach() # remove from backpropagation

        loss = loss_function(logits_strong, pseudolabels)
        
        return (loss * mask).mean()