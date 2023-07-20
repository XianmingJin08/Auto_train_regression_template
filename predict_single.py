import torch
import numpy as np
import pickle
from src.models import cnn_lstm, lstm, mlp  # Import your model class
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Define a function for data preprocessing
def preprocess_data(data,scaler_path):
    # Perform any necessary preprocessing on the input data
    N,T,K = data.shape
    data_reshaped = data.reshape(-1, T*K)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f) 
    preprocessed_data = scaler.transform(data_reshaped).reshape(-1, T, K)  # Preprocess the data here
    return preprocessed_data

# Define a function for loading the saved model
def load_model(model_path, model_type):
    if model_path.endswith('.pt'):
        model = model_type()
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif model_path.endswith('.pkl'):
        model = joblib.load(model_path)
    else:
        raise ValueError('Unsupported model file type. The file should be a .pt or .pkl file.')
    return model

# Define a function for making predictions
def predict(model, data, scaler_path):
    # Preprocess the input data
    data = data.reshape(1,1,len(data))
    preprocessed_data = preprocess_data(data,scaler_path)

    if isinstance(model, (cnn_lstm.CNNLSTMModel, lstm.LSTMModel, mlp.MLPModel)):
        # Convert the preprocessed data to a PyTorch tensor
        tensor_data = torch.Tensor(preprocessed_data)
        # Make the prediction
        with torch.no_grad():
            output = model(tensor_data)
        # Convert the output to the desired format (e.g., numpy array)
        prediction = output.numpy()
    else:
        # Prepare the data for a classic ML model
        classic_data = preprocessed_data.reshape(-1)
        prediction = model.predict([classic_data])

    return prediction

if __name__ == '__main__':
    x = np.array([114.9 ,  50.8 ,   3.75,  17.  ,  35., 39.9])
    # model_path = "saved_model_2023/regression_2023.pt"
    # model = load_model(model_path, lstm.LSTMModel)
    model_path = "saved_model_2023/catboost_all.pkl"
    model = load_model(model_path, CatBoostRegressor)    
    print(predict(model, x, "scaler.pkl"))