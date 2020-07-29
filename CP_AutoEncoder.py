""" This class provides functionality for building the customer profile (CP) autoencoder. It assumes that that the user
provides a normalized customer profile tensor. As of 7/2020, the CP autoencoder will handle a tensor of 40 features
for each customer. Our goal is to reduce that to around 10 features for the CNN to train on. """

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pylab
from torch import nn
from torch import optim
import torchvision
from sklearn import preprocessing

def combined_criterion(output_tensor, input_tensor, indices_for_mse, indices_for_ce):
    """ this is a tailored loss function. Accepts the original tensor and the decoded tensor. We also indicate
    which features should be evaluated on MSE Loss and BCE Loss by their indices """

    mse_loss = nn.MSELoss()(output_tensor[:,indices_for_mse], input_tensor[:,indices_for_mse])
    bce_loss = nn.BCELoss()(output_tensor[:, indices_for_ce], input_tensor[:, indices_for_ce]) ##need to identify the features that are to be fit to CE Loss
    loss = mse_loss + bce_loss
    return loss


def load_data(normalized_cp_tensor, split=True, train_test_split=0.8):
    """ Splits the cp tensor into a training and testing dataset. This test set will be used to measure
    the training performance of the autoencoder. Assume a normalized CP tensor is given. Returns dataloaders
    which will be used for training."""
    if split:
        subset_train_size = round(normalized_cp_tensor.shape[0] * train_test_split)
        subset_test_size = normalized_cp_tensor.shape[0] - subset_train_size
        train_data = torch.utils.data.random_split(train_tensor, [subset_train_size, subset_test_size])[0]
        test_data = torch.utils.data.random_split(train_tensor, [subset_train_size, subset_test_size])[1]
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4,
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4,
                                                   pin_memory=True)
        return train_loader, test_loader
    else:
        return torch.utils.data.DataLoader(normalized_cp_tensor, batch_size=128, shuffle=True, num_workers=4,
                                           pin_memory=True)

class CPAutoEncoder(nn.Module):
    def __init__(self, input_features=43,encoded_vector_length=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(CPAutoEncoder, self).__init__()
        self.input_features = input_features
        self.encoded_vector_length = encoded_vector_length
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 20),
            nn.ReLU(True),
            nn.Linear(20, encoded_vector_length),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(encoded_vector_length, 20),
            nn.ReLU(True),
            nn.Linear(20, input_features),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build(self):
        pass
        ## TODO build a the network architecture based on number of hidden layers and regularization desired


    def evaluate(self):
        pass
        ## evaluate the model performance on a dataset. Compare loss on original tensor to decoded tensor

    def get_embeddings(self):
        pass
        ## get the embedded tensor for given normalized data


def train_cp_autoencoder(epochs, model, train_loader, test_loader, optimizer, combined_criterion,
                         indices_for_mse, indices_for_ce, device):
    train_loss_tracker = []
    test_loss_tracker = []
    for epoch in range(epochs):
        loss = 0
        test_loss = 0
        # what is test loss?
        with torch.no_grad():
            model.train(False)
            #valid operations
            for test_batch in test_loader:
                batch_features = test_batch.view(-1, 43).to(device)
                model.eval()
                outputs = model(batch_features.float())
                batch_test_loss = combined_criterion(outputs.float(), batch_features.float(), indices_for_mse, indices_for_ce)
                optimizer.zero_grad()
                test_loss += batch_test_loss.item()

            test_loss = test_loss / len(test_loader)
            test_loss_tracker.append(test_loss)

        for batch_features in train_loader:
            model.train(True)

            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 43).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features.float())

            # compute training reconstruction loss
    #         train_loss = criterion(outputs, batch_features)
            train_loss = combined_criterion(outputs, batch_features, indices_for_mse, indices_for_ce)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)
        train_loss_tracker.append(loss)

        # display the epoch training loss
        print("epoch : {}/{}, training loss = {:.6f}".format(epoch + 1, epochs, loss))
    return model



df = pd.read_csv('postNormalizedf2.csv', index_col=0)
print(df.head())

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train_tensor = torch.tensor(train.values, dtype=torch.float32)

load_data(train_tensor, True, 0.7)

