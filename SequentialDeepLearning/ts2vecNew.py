import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve, auc, average_precision_score
import csv

#GL Support Functions
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

def predictionAnalysis(yTrue, yPred):
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

class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        modelWeightsFolder=None,
        modelName=None,
        fold=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

        self.modelWeightsFolder = modelWeightsFolder
        self.modelName = modelName
        self.fold = fold
    
    def fit(self, train_loader, test_loader, yTrain, yTest, n_epochs=10, verbose=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []

        prev_test_AUC_ROC = None
        num_decreases = 0
        
        modelMetrics = []

        for epoch in range(n_epochs):
            cum_loss = 0
            n_batches = 0
            for batch in train_loader.generator():                
                x = batch[0].to(self.device)
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_batches += 1

            if verbose:
                print(f"Epoch #{epoch}: loss={cum_loss}")
            loss_log.append(cum_loss / n_batches)

            if epoch % 2 == 0:
                with torch.no_grad():
                    train_embeddings = self.encode_with_dataloader(train_loader, encoding_window="full_series")
                    test_embeddings = self.encode_with_dataloader(test_loader, encoding_window="full_series")
                    classifier = LogisticRegression(class_weight = 'balanced', max_iter=1000)
                    classifier.fit(train_embeddings, yTrain)
                    yTrainPred = classifier.predict_proba(train_embeddings)[:,1]
                    yTestPred = classifier.predict_proba(test_embeddings)[:, 1]
                    epochTrainLogLoss = getLogLoss(yTrain, yTrainPred)
                    epochTestLogLoss = getLogLoss(yTest, yTestPred)

                    train_fpr, train_tpr, train_thresholds = roc_curve(yTrain, yTrainPred, pos_label=1)
                    train_AUC_ROC = auc(train_fpr, train_tpr)
                    train_AUC_PR = average_precision_score(yTrain, yTrainPred)
                    test_fpr, test_tpr, test_thresholds = roc_curve(yTest, yTestPred, pos_label=1)
                    test_AUC_ROC = auc(test_fpr, test_tpr)
                    test_AUC_PR = average_precision_score(yTest, yTestPred)

                    print("\n{} Epoch {}\n Loss: {:.4f},{:.4f}\nAUC-ROC: {:.4f},{:.4f}\nAUC-PR: {:.4f},{:.4f}".format\
             ("FOLD", epoch, epochTrainLogLoss, epochTestLogLoss, train_AUC_ROC, test_AUC_ROC, train_AUC_PR, test_AUC_PR), flush=True)
                    predictionAnalysis(yTrain, yTrainPred)
                    predictionAnalysis(yTest, yTestPred)

                    modelMetrics.append({
                        'fold': self.fold,
                        'epoch': epoch,
                        'epochTrainLogLoss': epochTrainLogLoss,
                        'epochTestLogLoss': epochTestLogLoss,
                        'train_AUC_ROC': train_AUC_ROC,
                        'test_AUC_ROC': test_AUC_ROC,
                        'train_AUC_PR': train_AUC_PR,
                        'test_AUC_PR': test_AUC_PR
                    })

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
        return loss_log, modelMetrics, self.net.state_dict()
    
    def fit_original(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode_with_dataloader(self, dataloader, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0):
        ''' Compute representations using the model with a DataLoader for batching.

        Args:
            dataloader (DataLoader): PyTorch DataLoader containing the input data.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.

        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'

        org_training = self.net.training
        self.net.eval()

        with torch.no_grad():
            output = []
            for batch in dataloader.generator():
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    for i in range(0, x.size(1), sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, x.size(1))],
                            left=-l if l < 0 else 0,
                            right=r - x.size(1) if r > x.size(1) else 0,
                            dim=1
                        )
                        out = self._eval_with_pooling(
                            x_sliding,
                            mask,
                            slicing=slice(sliding_padding, sliding_padding + sliding_length),
                            encoding_window=encoding_window
                        )
                        reprs.append(out)

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()




    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
