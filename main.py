import torch
import numpy as np
import json
from src import data, train, utils
from src.models import cnn_lstm, lstm, mlp  # Import your models here
import os 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
def read_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    return config



class ModelManager:

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        self.train_mode = "continuous" if config['model']['mode'] == "continuous" else "scratch"
        self.model_dict = {
            'cnn_lstm': cnn_lstm.CNNLSTMModel(), 
            'lstm': lstm.LSTMModel(),
            'mlp': mlp.MLPModel(), 
            'random_forest': RandomForestRegressor(), 
            'linear_regression': LinearRegression(), 
            'svm': SVR(),
            'gradient_boosting': GradientBoostingRegressor(),
            'xgboost': XGBRegressor(),
            'lightgbm': LGBMRegressor(),
            'catboost': CatBoostRegressor(),
            'elasticnet': ElasticNet()
        }

        # Initialize Hyperparameters
        self.n_epochs = 100
        self.learning_rate = 0.001
        self.patience = 20
        self.batch_size = 32
        
        self.train_data_path = "data/data_A.npy"
        self.labels_path = "data/labels.npy"
        self.scaler_save_path = None
        # Update hyperparameters based on config
        self.update_hyperparameters()
        
        
    def update_hyperparameters(self):
        """
        Helper function to replace the initialized hyperparameters 
        with the ones specified in the config file. If the hyperparameter
        does not exist in the config file, keep the initialized value.
        """
        self.n_epochs = self.config.get('hyperparameters', {}).get('n_epochs', self.n_epochs)
        self.learning_rate = self.config.get('hyperparameters', {}).get('learning_rate', self.learning_rate)
        self.patience = self.config.get('hyperparameters', {}).get('patience', self.patience)
        self.batch_size = self.config.get('hyperparameters', {}).get('batch_size', self.batch_size)

        self.train_data_path = self.config.get('dataset_path', {}).get('train_data_path', self.train_data_path)
        self.labels_path = self.config.get('dataset_path', {}).get('labels_path', self.labels_path)
        self.scaler_save_path = self.config.get('scalar', {}).get('save_path', self.scaler_save_path)


    def preprocess_data(self, X):
        # Reshape the data to the format required by classic ML models
        return X.reshape(X.shape[0], -1)

    def load_data(self):
        # Update the data loading method to return the datasets directly
        train_data = utils.normalize_data_scaler(np.load(self.train_data_path,allow_pickle=True), load_path=None,
                                                 save_path=self.scaler_save_path)
        labels = np.load(self.labels_path,allow_pickle=True)

        # Split the data into training, validation, and testing sets
        X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_classic_model(self, model, X_train, y_train):
        # Train the model
        model.fit(X_train, y_train)
        return model

    def validate_classic_model(self, model, X_val, y_val):
        # Compute the validation MSE
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        return val_mse, val_mae

    def test_classic_model(self, model, X_test, y_test):
        # Compute the test MSE
        test_mse = mean_squared_error(y_test, model.predict(X_test))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        return test_mse, test_mae

    def get_model(self, model_name):
        return self.model_dict[model_name]

    def initialize(self, model):
        return utils.initialize_model(model, self.device, learning_rate=self.learning_rate)

    def train_model(self, model, optimizer, train_loader, valid_loader):
        # Initialize early stopping
        early_stopping = utils.EarlyStopping(patience=self.patience, verbose=True)
            
        for epoch in range(self.n_epochs):
            train_loss = train.train(model, self.device, train_loader, optimizer)
            print(f'Epoch: {epoch}, Train Loss: {train_loss}')
            val_loss, val_mse, _ = train.validate(model, self.device, valid_loader)
            print(f'Epoch: {epoch}, Val Loss: {val_loss}, Val MSE: {val_mse}')

            # Early stopping
            early_stopping(val_mse, model, self.config['model']['save_path'])
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return model, val_mse

    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        best_mse = 1e9
        best_model = None
        best_model_path = None

        # Create data loaders for deep learning models
        train_loader, valid_loader, test_loader = data.get_dataloaders(X_train, y_train, batch_size=self.batch_size)

        # Train each model
        for model_name, model in self.model_dict.items():
            print(f"Training model: {model_name}")

            # Check if the model is a deep learning model or a classic ML model
            if isinstance(model, (cnn_lstm.CNNLSTMModel, lstm.LSTMModel, mlp.MLPModel)):
                # Initialize the model and optimizer
                model, optimizer = self.initialize(model)

                # Train the model
                model, val_mse = self.train_model(model, optimizer, train_loader, valid_loader)

                test_mse = self.test_model(model, test_loader, X_test, y_test)

            else:
                # Preprocess the data for classic ML models
                X_train_classic = X_train.reshape(X_train.shape[0], -1)
                X_val_classic = X_val.reshape(X_val.shape[0], -1)
                X_test_classic = X_test.reshape(X_test.shape[0], -1)

                # Train, validate, and test the classic ML model
                model = self.train_classic_model(model, X_train_classic, y_train)
                val_mse, val_mae = self.validate_classic_model(model, X_val_classic, y_val)
                test_mse, test_mae = self.test_classic_model(model, X_test_classic, y_test)
                print ("The model is: ", type(model).__name__)
                print ("the valid mse and test mse are {} {}".format(val_mse,test_mse))
                print ("the valid mae and test mae are {} {}".format(val_mae,test_mae))
            # Compare with best model
            if val_mse < best_mse:
                best_mse = val_mse
                print ("best model changed from {} to {}".format(type(best_model).__name__, model_name))
                best_model = model
                # Save classic ML models as .pkl and deep learning models as .pt
                if isinstance(best_model, (cnn_lstm.CNNLSTMModel, lstm.LSTMModel, mlp.MLPModel)):
                    best_model_path = f"saved_model_2023/{model_name}.pt"
                    torch.save(best_model.state_dict(), best_model_path)
                else:
                    best_model_path = f"saved_model_2023/{model_name}.pkl"
                    joblib.dump(best_model, best_model_path)


        print ("The best model is: ", type(best_model).__name__)
        return best_model, best_model_path   

    def load_model(self, model, model_path):
        return train.load_model(model, model_path)
    
    def test_model(self, model, test_loader, X_test, y_test):
        if isinstance(model, (cnn_lstm.CNNLSTMModel, lstm.LSTMModel, mlp.MLPModel)):
            test_mse = train.test(model, self.device, test_loader)
            print('Test MSE:', test_mse)
        else:
            X_test_classic = X_test.reshape(X_test.shape[0], -1)
            test_mse, test_mae = self.test_classic_model(model, X_test_classic, y_test)
            print('Test MSE:', test_mse)
            print('Test MAE:', test_mae)
        return test_mse

def main():
    # Read config
    config = read_config('config.json')

    model_manager = ModelManager(config)

    X_train, y_train, X_val, y_val, X_test, y_test = model_manager.load_data()

    # Create data loaders for deep learning models
    train_loader, valid_loader, test_loader = data.get_dataloaders(X_train, y_train, batch_size=model_manager.batch_size)

    # Choose a specific model or train all models and save the best
    if config['model']['name'] != "all":
        model = model_manager.get_model(config['model']['name'])
        # Check if the model is a deep learning model or a classic ML model
        if isinstance(model, (cnn_lstm.CNNLSTMModel, lstm.LSTMModel, mlp.MLPModel)):
            # If continuous learning, load current model
            if config['model']['mode'] == "continuous":
                model = model_manager.load_model(model, config['model']['load_path'])
                print ("load the model from ", config['model']['load_path'])
            # Initialize the model and optimizer
            model, optimizer = model_manager.initialize(model)

            # Train the model
            model, _ = model_manager.train_model(model, optimizer, train_loader, valid_loader)
            # Test the model
            model_manager.test_model(model, test_loader, X_test, y_test)

        else:
            # Preprocess the data for classic ML models
            X_train_classic = X_train.reshape(X_train.shape[0], -1)
            X_val_classic = X_val.reshape(X_val.shape[0], -1)
            X_test_classic = X_test.reshape(X_test.shape[0], -1)

            # Train, validate, and test the classic ML model
            model = model_manager.train_classic_model(model, X_train_classic, y_train)
            val_mse,val_mae = model_manager.validate_classic_model(model, X_val_classic, y_val)
            test_mse,test_mae = model_manager.test_classic_model(model, X_test_classic, y_test)
            print ("The model is: ", type(model).__name__)
            print ("the valid mse and test mae are {} {}".format(val_mse,test_mse))
            print ("the valid mae and test mae are {} {}".format(val_mae,test_mae))
            # Save the trained model
            joblib.dump(model, f"saved_model_2023/{config['model']['name']}.pkl")

    else:
        # Train all models and save the best
        model, _ = model_manager.train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Test the model
    model_manager.test_model(model, test_loader, X_test, y_test)

    # model_manager = ModelManager(config)
    # X_train, y_train, X_val, y_val, X_test, y_test = model_manager.load_data()
    # # Combine all data for final training
    # X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    # y_all = np.concatenate((y_train, y_val, y_test), axis=0)
    # X_all_classic = X_all.reshape(X_all.shape[0], -1)
    # model = CatBoostRegressor()
    # model.fit(X_all_classic,y_all)
    # best_model_path = f"saved_model_2023/catboost_all.pkl"
    # joblib.dump(model, best_model_path)
if __name__ == '__main__':
    main()
