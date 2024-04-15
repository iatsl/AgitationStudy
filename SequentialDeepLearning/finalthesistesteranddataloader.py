# -*- coding: utf-8 -*-

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_curve, auc, average_precision_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random
import os
from tqdm import tqdm
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


xArrayFilePath = #insert filepath to sequence data h5 file
yArrayFilePath = #insert filepath to label data h5 file
indicesFilePath = #insert filepath to indices file path specifying fold 1, fold 2 split
modelWeightsFolder = #insert filepath to folder where weights and results will be saved
dataNormalizationPath = #insert filepath to file with mean and std dev for each fold for data normalization

class HDF5DataLoader:
    def __init__(self, X_file_path, Y_file_path, datasetIndices, batch_size, shuffle=True, downsample_factor = 1, normalize=False, datasetMean = None, datasetStd = None):
        self.datasetIndices = datasetIndices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X_h5_file = h5py.File(X_file_path, 'r')
        self.Y_h5_file = h5py.File(Y_file_path, 'r')
        self.num_samples = len(self.datasetIndices)
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.datasetMean = datasetMean
        self.datasetStd = datasetStd
        if self.shuffle:
            np.random.shuffle(self.datasetIndices)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generator(self):
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.datasetIndices[i:i + self.batch_size]
            batch_indices.sort() #indices need to be in increasing order to read from h5 file
            batch_data = self.X_h5_file['masterX'][batch_indices]  
            batch_labels = self.Y_h5_file['masterY'][batch_indices] 
            if self.downsample_factor > 1:
                batch_data = self.downsample(batch_data)
            if self.normalize and self.datasetMean is not None and self.datasetStd is not None:
                batch_data = self.normalizeBatch(batch_data, self.datasetMean, self.datasetStd)
            batch_data = torch.from_numpy(batch_data).float().to(self.device)
            batch_labels = torch.from_numpy(batch_labels).float().to(self.device)
            yield batch_data, batch_labels

    def downsample(self, batch_data):
        """Function to perform random downsampling of sequences from batch data
        """
        num_timesteps = batch_data.shape[1]
        downsampled_timesteps = num_timesteps // self.downsample_factor
        downsampled_batch_data = np.zeros((batch_data.shape[0], downsampled_timesteps, batch_data.shape[2]))

        for i in range(batch_data.shape[0]):
            indices = np.random.choice(num_timesteps, downsampled_timesteps, replace=False)
            downsampled_batch_data[i] = batch_data[i, indices]

        return downsampled_batch_data

    def normalizeBatch(self, batch_data, datasetMean, datasetStd):
        """Function to normalize all sequences in batch
        """
        datasetMean = np.tile(datasetMean, (batch_data.shape[0], batch_data.shape[1], 1))
        datasetStd = np.tile(datasetStd, (batch_data.shape[0], batch_data.shape[1], 1))
        batch_data = (batch_data - datasetMean) / datasetStd
        return batch_data

    def close(self):
        self.X_h5_file.close()
        self.Y_h5_file.close()

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class Transformer(nn.Module):
    """Transformer model implementation"""
    def __init__(self, input_channels, hidden_size, num_layers, num_heads):
        super(Transformer, self).__init__()
        # Input embedding layer
        self.embedding = nn.Linear(input_channels, hidden_size)
        # Positional encoding
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoding = PositionalEncoding(hidden_size).to(device)
        # Transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads),
            num_layers
        )
        # Pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)  # Single-variable output

    def forward(self, x):
        # Input embedding
        x = self.embedding(x)
        # Positional encoding
        x = self.positional_encoding(x)
        # Transformer encoder layers
        x = self.transformer_encoder(x)
        # Pooling
        x = x.transpose(1, 2)  # Transpose for 1D pooling
        x = self.pooling(x).squeeze(-1)
        # Output layer
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model to integrate sequencing information into embeddings"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        encoding = self.encoding.to(x.device)
        return x + encoding[:, :x.size(1)].detach()

class LSTM(nn.Module):
    """LSTM model implementation"""
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, dropout=0.0):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])  # Use the last time step's output for classification
        out = torch.sigmoid(out)
        return out

class TCNBlock(nn.Module):
    """TCN Block implementation"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return out[:, :, :residual.size(2)] + residual  # Trim the output to match the residual size

class TCN(nn.Module):
    """TCN Model implementation"""
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, dilation)]
            layers += [nn.Dropout(dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, seq_length)
        x = self.tcn(x)
        x = x[:, :, -1]  # Take the last output for classification
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


class AgitationModel:
    def __init__(self, modelName, batchSize = 256, epochs = 10, \
                 currentFold = 0, currentEpoch = 0, loadModel = False, modelWeightsPath = None):
        self.modelName = modelName
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.batchSize = batchSize
        self.epochs = epochs
        self.history = {'fold0': {'trainLoss': [], 'testLoss': [], 'AUC': []},
                        'fold1': {'trainLoss': [], 'testLoss': [], 'AUC': []}}
        self.CVTestLoss = None
        self.CVAUC = None
        self.modelWeightsFolder = modelWeightsFolder
        self.currentFold = currentFold
        self.currentEpoch = currentEpoch

        if loadModel:
            self.createModelTransformer()
            self.loadModelTransformer(modelWeightsPath)
        else:
            self.createModelTransformer()

    def loadModel(self, modelWeightsPath):
        """Old version for tensorflow, use loadModelTransformer() for pytorch instead"""
        try:
            # Attempt to load the model from the specified file path
            self.model = load_model(modelWeightsPath)
            print(f'Model loaded from {modelWeightsPath}')
        except (OSError, IOError):
            # If loading fails, create a new model
            print(f'Failed to load model from {modelWeightsPath}')

    def loadModelTransformer(self, modelWeightsPath):
        state_dict = torch.load(modelWeightsPath)
        self.model.load_state_dict(state_dict)
        print(f'Model loaded from {modelWeightsPath}')
        return

    def createModelTransformer(self):
        """Define your model here. Not only for Transformer.
        
        Models used in Thesis
            model = Transformer(input_channels=6, hidden_size=32, num_layers=2, num_heads=4).to(device)
            model = LSTM(input_size=6, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2).to(device)
            model = TCN(input_size=6, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2).to(device)
        """
        #model = Transformer(input_channels=6, hidden_size=32, num_layers=2, num_heads=4).to(device)
        #model = LSTM(input_size=6, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2).to(device)
        model = TCN(input_size=6, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2).to(device)
        criterion = nn.BCELoss(weight=None)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        return

def getLogLoss(yTest, yTestPredModel):
    """
    Calculate the log loss between true labels (yTest) and predicted probabilities (yTestPredModel).

    Parameters:
    - yTest: numpy array, true labels (ground truth)
    - yTestPredModel: numpy array, predicted probabilities for each class

    Returns:
    - float, log loss
    """
    epsilon = 1e-7  # Choose an appropriate epsilon for float16
    yTestPredModel = np.clip(yTestPredModel, epsilon, 1 - epsilon)
    log_loss = -np.mean(yTest * np.log(yTestPredModel) + (1 - yTest) * np.log(1 - yTestPredModel))
    if log_loss != log_loss:
      print(yTestPredModel)
      print(yTest)
    return log_loss

def getConfusionMatrix(yTest, yTestPredModel):
    tp = np.sum(np.logical_and(yTest == 1, yTestPredModel == 1))
    tn = np.sum(np.logical_and(yTest == 0, yTestPredModel == 0))
    fp = np.sum(np.logical_and(yTest == 0, yTestPredModel == 1))
    fn = np.sum(np.logical_and(yTest == 1, yTestPredModel == 0))

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return confusion_matrix

def plotAUC(fpr, tpr, AUC):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def predictionAnalysis(yTrue, yPred):
    """
    Show mean and standard deviations of predictions where yTrue = 1 and yTrue = 0
    """
    mean_y1 = np.mean(yPred[yTrue == 1])
    std_y1 = np.std(yPred[yTrue == 1])
    mean_y0 = np.mean(yPred[yTrue == 0])
    std_y0 = np.std(yPred[yTrue == 0])
    print("Mean and std deviation when yTrain = 1: Mean: {:.4f}, Std Deviation: {:.4f}".format(mean_y1, std_y1), flush=True)
    print("Mean and std deviation when yTrain = 0: Mean: {:.4f}, Std Deviation: {:.4f}".format(mean_y0, std_y0), flush=True)

def write_metrics_to_csv(filename, fold, epoch, train_log_loss, test_log_loss, train_auc_roc, test_auc_roc, train_auc_pr, test_auc_pr):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([fold, epoch, train_log_loss, test_log_loss, train_auc_roc, test_auc_roc, train_auc_pr, test_auc_pr])

def trainEvaluateAgitationModel(AgitationModel, trainIndices, testIndices, fold, datasetMean, datasetStd, classWeight = False):
    """Performs one fold of training. 
    Specify which indices are to be used as train and test as well as normalization based on training set
    """
    #Load y values based on train and test indices
    with h5py.File(yArrayFilePath, 'r') as hf:
        yArray = hf['masterY'][:]
    yTrain = yArray[trainIndices]
    yTest = yArray[testIndices]



    #Create dataloaders
    trainDataLoader = HDF5DataLoader(xArrayFilePath, yArrayFilePath, trainIndices, batch_size=AgitationModel.batchSize, downsample_factor = 8, shuffle=False, normalize=True, datasetMean = datasetMean, datasetStd = datasetStd)
    testDataLoader = HDF5DataLoader(xArrayFilePath, yArrayFilePath, testIndices, batch_size=AgitationModel.batchSize, downsample_factor = 8, shuffle=False, normalize=True, datasetMean = datasetMean, datasetStd = datasetStd)

    if classWeight:
        cW = class_weight.compute_class_weight('balanced', classes=np.unique(yTrain), y=yTrain)
        cW = cW / np.sum(cW)

    # Initialize variables to track AUC-ROC for early stopping
    prev_test_AUC_ROC = None
    num_decreases = 0

    for epoch in range(AgitationModel.currentEpoch, AgitationModel.epochs):
        train_loader_tqdm = tqdm(trainDataLoader.generator(), total=len(trainDataLoader))
        for batchIdx, batch in enumerate(train_loader_tqdm):
            data_batch, label_batch = batch[0].to(device), batch[1].to(device)

            # Training step

            AgitationModel.optimizer.zero_grad()
            output = AgitationModel.model(data_batch)
            batchWeights = torch.where(label_batch == 1, torch.full_like(label_batch, cW[1]), torch.full_like(label_batch, cW[0])).unsqueeze(1)
            AgitationModel.criterion.weight = batchWeights
            loss = AgitationModel.criterion(output, label_batch.unsqueeze(1))
            loss.backward()
            AgitationModel.optimizer.step()
            train_loader_tqdm.set_description(f"Epoch {epoch} Batch {batchIdx}")
            train_loader_tqdm.set_postfix(loss=loss.item())

        if epoch%2==0 or epoch==AgitationModel.epochs-1:
            yTrainPred = batchPredictAllTransformer(AgitationModel.model, trainDataLoader, datasetMean, datasetStd)
            epochTrainLogLoss = getLogLoss(yTrain, yTrainPred)

            yTestPred = batchPredictAllTransformer(AgitationModel.model, testDataLoader, datasetMean, datasetStd)
            epochTestLogLoss = getLogLoss(yTest, yTestPred)

            train_fpr, train_tpr, train_thresholds = roc_curve(yTrain, yTrainPred, pos_label=1)
            train_AUC_ROC = auc(train_fpr, train_tpr)
            train_AUC_PR = average_precision_score(yTrain, yTrainPred)
            test_fpr, test_tpr, test_thresholds = roc_curve(yTest, yTestPred, pos_label=1)
            test_AUC_ROC = auc(test_fpr, test_tpr)
            test_AUC_PR = average_precision_score(yTest, yTestPred)

            #Print Metrics
            print("\n{} Epoch {}\n Loss: {:.4f},{:.4f}\nAUC-ROC: {:.4f},{:.4f}\nAUC-PR: {:.4f},{:.4f}".format\
             (fold, epoch, epochTrainLogLoss, epochTestLogLoss, train_AUC_ROC, test_AUC_ROC, train_AUC_PR, test_AUC_PR), flush=True)
            plotAUC(train_fpr, train_tpr, train_AUC_ROC)
            plotAUC(test_fpr, test_tpr, test_AUC_ROC)
            predictionAnalysis(yTrain, yTrainPred)
            predictionAnalysis(yTest, yTestPred)

            #Write Metrics to File
            resultsFilePath = AgitationModel.modelWeightsFolder+'/{}_{}_results.csv'.format(AgitationModel.modelName, fold)
            write_metrics_to_csv(resultsFilePath, fold, epoch, epochTrainLogLoss, epochTestLogLoss, train_AUC_ROC, test_AUC_ROC, train_AUC_PR, test_AUC_PR)

            #Early Stopping
            if prev_test_AUC_ROC is not None:
                if test_AUC_ROC < prev_test_AUC_ROC:
                    num_decreases += 1
                    if num_decreases == 2:
                        print("Early stopping triggered.")
                        break
                else:
                    num_decreases = 0

            prev_test_AUC_ROC = test_AUC_ROC

        #Save model weights
        torch.save(AgitationModel.model.state_dict(), AgitationModel.modelWeightsFolder+'/{}_{}_weights.pth'.format(AgitationModel.modelName, fold))
        AgitationModel.currentEpoch = epoch+1

    return


def batchPredictAllTransformer(model, dataLoader, datasetMean, datasetStd):
    model.eval()
    all_predictions = []

    # Iterate through batches and accumulate predictions
    data_loader_tqdm = tqdm(dataLoader.generator(), total=len(dataLoader))
    for batchIdx, batch in enumerate(data_loader_tqdm):
        data_batch, label_batch = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():  # Disable gradient calculation during inference
            predictions = model(data_batch)
        predictions = predictions.cpu().numpy().flatten()
        all_predictions.append(predictions)
        data_loader_tqdm.set_description(f"Batch {batchIdx}")
    all_predictions = np.concatenate(all_predictions)
    model.train()
    return all_predictions

def agitationModelCrossValidation(AgitationModel):
    """Function to perform 2-Fold Cross Validation"""
    trainTestArray = scipy.io.loadmat(indicesFilePath)['teIdx'][:,0]
    #trainTestArray = generateTrainTestArray(len(xArray))

    with h5py.File(dataNormalizationPath, 'r') as hf:
        trainMean = hf['trainMean'][:]
        trainStd = hf['trainStd'][:]
        testMean = hf['testMean'][:]
        testStd = hf['testStd'][:]

    if AgitationModel.currentFold == 0:
        #1st Fold
        trainIndices = [index for index, value in enumerate(trainTestArray) if value == 0]
        testIndices = [index for index, value in enumerate(trainTestArray) if value == 1]
        trainEvaluateAgitationModel(AgitationModel, trainIndices, testIndices, 'fold0', trainMean, trainStd, classWeight=True)

        #Reset weights of model before second fold
        AgitationModel.createModelTransformer()
        AgitationModel.currentFold = 1
        AgitationModel.currentEpoch = 0


    #2nd Fold. Reverse the indexing between training set indices and test set indices
    trainIndices = [index for index, value in enumerate(trainTestArray) if value == 1]
    testIndices = [index for index, value in enumerate(trainTestArray) if value == 0]
    trainEvaluateAgitationModel(AgitationModel, trainIndices, testIndices, 'fold1', testMean, testStd, classWeight = True)
    return

def main():
    #myAgitationModel = AgitationModel("Transformer_ds8_final", epochs = 50, batchSize = 512, loadModel = False)
    myAgitationModel = AgitationModel("TCN_ds8_final", epochs=50, batchSize=512, loadModel = False)
    agitationModelCrossValidation(myAgitationModel)



if __name__ == '__main__':
    main()