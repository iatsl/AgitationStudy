# -*- coding: utf-8 -*-

import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_curve, auc
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
import torch.nn.functional as F
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

xArrayFilePath = #insert filepath to sequence data h5 file
yArrayFilePath = #insert filepath to label data h5 file
indicesFilePath = #insert filepath to indices file path specifying fold 1, fold 2 split
modelWeightsFolder = #insert filepath to folder where weights and results will be saved
dataNormalizationPath = #insert filepath to file with mean and std dev for each fold for data normalization

#Download TS2Vec Library from https://github.com/zhihanyue/ts2vec
#Note ts2vec.py has been modified and renamed ts2vecNew.py for the agitation study training
from ts2vecNew import TS2Vec

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            batch_data = torch.tensor(batch_data, dtype = torch.float32).to(self.device)
            batch_labels = torch.tensor(batch_labels, dtype = torch.float32).to(self.device)
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

folds = ['fold0', 'fold1']
downsample_factor = 8

for fold in folds:
    #Load TS2Vec model
    ts2vec_model = TS2Vec(
        input_dims=6,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        modelWeightsFolder = modelWeightsFolder,
        modelName = 'TS2Vec_ds8_final',
        fold=fold
    )

    trainTestArray = scipy.io.loadmat(indicesFilePath)['teIdx'][:,0]

    with h5py.File(dataNormalizationPath, 'r') as hf:
        trainMean = hf['trainMean'][:]
        trainStd = hf['trainStd'][:]
        testMean = hf['testMean'][:]
        testStd = hf['testStd'][:]

    if fold == 'fold0':
        trainIndices = [index for index, value in enumerate(trainTestArray) if value == 0]
        testIndices = [index for index, value in enumerate(trainTestArray) if value == 1]
        datasetMean = trainMean
        datasetStd = trainStd
    elif fold == 'fold1':
        trainIndices = [index for index, value in enumerate(trainTestArray) if value == 0]
        testIndices = [index for index, value in enumerate(trainTestArray) if value == 1]
        datasetMean = testMean
        datasetStd = testStd

    batchSize = 256

    trainDataLoader = HDF5DataLoader(xArrayFilePath, yArrayFilePath, trainIndices, batch_size=batchSize, downsample_factor = downsample_factor, shuffle=False, normalize = True, datasetMean = datasetMean, datasetStd = datasetStd)
    testDataLoader = HDF5DataLoader(xArrayFilePath, yArrayFilePath, testIndices, batch_size=batchSize, downsample_factor = downsample_factor, shuffle=False, normalize = True, datasetMean = datasetMean, datasetStd = datasetStd)

    with h5py.File(yArrayFilePath, 'r') as hf:
        yArray = hf['masterY'][:]
    yTrain = yArray[trainIndices]
    yTest = yArray[testIndices]

    #Run modified TS2Vec training loop
    loss_log, modelMetrics, stateDict = ts2vec_model.fit(trainDataLoader, testDataLoader,  yTrain, yTest, n_epochs=50, verbose=True)

    #Save the model
    torch.save(stateDict, ts2vec_model.modelWeightsFolder+'/{}_{}_weights.pth'.format(ts2vec_model.modelName, ts2vec_model.fold))


    # Define the file path for the CSV file
    csv_file_path = ts2vec_model.modelWeightsFolder+'/{}_{}_results.csv'.format(ts2vec_model.modelName, ts2vec_model.fold)

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write each result to the CSV file
        for result in modelMetrics:
            writer.writerow([result['fold'], result['epoch'], result['epochTrainLogLoss'], result['epochTestLogLoss'],
                            result['train_AUC_ROC'], result['test_AUC_ROC'], result['train_AUC_PR'], result['test_AUC_PR']])