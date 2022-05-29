import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import Module, LSTM, Linear, MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class CryptoDataset(Dataset):
  """ Class to use for DataLoader """
  def __init__(self, X, y):
    self.X = X
    self.y = y
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return [self.X[idx], self.y[idx]]


class LSTMRNN(Module):
    """
    Long Short-Term Memory Recurrent Neural Network.
    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTMRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, batch_first=True)
        self.fc = Linear(hidden_size, num_classes)
        
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate RNN
        _, (h_out, _) = self.lstm(x, (h0, c0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


class LSTMBackend():
    def __init__(self, data_path, input_size: int = 1, num_classes: int = 1):
        self.input_size = input_size
        self.num_classes = num_classes
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_feather(data_path)


    def load_data(self, batch_size: int=2000):
        self.df['close'] = self.scaler.fit_transform(self.df['close'].values.reshape(-1, 1))
        # Split into X sequence and y
        X, y = [], []
        # Split into rolling windows of 110 hours
        L = 110 # Sequence Length, evenly divides 41,470
        for i in range(self.df.shape[0]-L):
            X.append(self.df.iloc[i:(i+L)].close)
            y.append(self.df.iloc[i+L].close)


        # --- TRAIN / VAL / TEST split --- #
        X = np.array(X)
        y = np.array(y)

        train_size = int((1-0.33) * y.shape[0])
        val_size = train_size - int(0.2 * train_size)

        # Training is from start until validation index
        X_train = X[:val_size]
        y_train = y[:val_size]
        # Validation is from validation index till end of training index
        X_val = X[val_size:train_size]
        y_val = y[val_size:train_size]
        # Testing is from training index until end
        X_test = X[train_size:]
        y_test = y[train_size:]

        # ---- Transform to ()num_layers, X.shape[0], hidden_state) shape ---- #
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], 1)

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], 1)

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        X_all = X.reshape(X.shape[0], X.shape[1], 1)
        y_all = y.reshape(y.shape[0], 1)

        train_dl = DataLoader(
            CryptoDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        validation_dl = DataLoader(
            CryptoDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        test_dl = DataLoader(
            CryptoDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        full_dl = DataLoader(
            CryptoDataset(X_all,y_all),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.test_dl = test_dl
        self.full_dl = full_dl
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_all = X_all
        self.y_all = y_all

    


    def finish(self):
        wandb.finish()

    def calculate_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Re-scale back to original values, and calculate the mean of percentage differences.

        Args:
            y_pred (`torch.Tensor`): predicted values
            y_true (`torch.Tensor`): true values
        
        Returns:
            `float`: mean percentage difference
        """
        _pred_scaled = self.scaler.inverse_transform(y_pred.cpu().data.numpy())
        _true_scaled = self.scaler.inverse_transform(y_true.cpu().data.numpy())
        return np.mean(1 - abs((_pred_scaled-_true_scaled)/_true_scaled))


    def train_model(self, config=None, model=None):
        with wandb.init(config=config):
            config = wandb.config
            if model is None:
                model = LSTMRNN(self.num_classes, self.input_size, config.hidden_layers, config.num_layers)
            # Loss Function
            criterion = MSELoss()
            # Monitor Gradients
            wandb.watch(model, log_freq=100)
            # Move model to `self.device`
            model.to(self.device)

            optimizer = Adam(model.parameters(), lr=config['learning_rate'])
            for _ in tqdm(range(config['epochs'])):
                model.train() # Set state to training
                for _, (inputs, targets) in enumerate(self.train_dl):
                    # Move to `self.device` and convert to `float`
                    inputs = inputs.to(self.device).float()
                    targets = targets.to(self.device).float()
                    optimizer.zero_grad() # Clear gradients
                    yhat = model(inputs)
                    loss = criterion(yhat, targets)
                    _acc = self.calculate_accuracy(yhat, targets)

                    # To Log to Weights & Biases
                    train_metrics = {
                        'train/train_loss': loss.item(),
                        'train/train_accuracy': _acc
                    }

                    # Backpropogate
                    loss.backward()
                    optimizer.step()
                
                model.eval() # Set state to inference
                with torch.no_grad():
                    for _, (inputs, targets) in enumerate(self.validation_dl):
                        # Move to `self.device` and convert to `float`
                        inputs = inputs.to(self.device).float()
                        targets = targets.to(self.device).float()
                        yhat = model(inputs)
                        loss = criterion(yhat, targets)
                        _acc = self.calculate_accuracy(yhat, targets)

                        # To Log to Weights & Biases
                        val_metrics = {
                        'val/val_loss': loss.item(),
                        'val/val_accuracy': _acc
                    }
                
                # Log
                wandb.log({**train_metrics, **val_metrics})
        return model

    def evaluate_model(self, model, test_dl):
        predictions, accuracy = [], []
        model.eval()
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(test_dl):
                # Move to `device` and convert to `float`
                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()
                yhat = model(inputs)
                _acc = self.calculate_accuracy(yhat, targets)
                accuracy.append(_acc)
                predictions.extend(yhat.cpu().numpy())
        return np.array(predictions), np.mean(accuracy)
