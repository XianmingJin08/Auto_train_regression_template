import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
from predict_single import predict, load_model
from src.models import lstm
# Define a function for data preprocessing
def preprocess_data(data,scaler_path):
    # Perform any necessary preprocessing on the input data
    N,T,K = data.shape
    data_reshaped = data.reshape(-1, T*K)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f) 
    preprocessed_data = scaler.transform(data_reshaped).reshape(-1, T, K)  # Preprocess the data here
    return preprocessed_data

def recommend_adjustments(X, y, x_input, y_low, y_upper, scaler_path, fix_idxes=None):
    # normalize X and x_input
    N,T,K = X.shape
    X_norm = preprocess_data(X,scaler_path=scaler_path)
    x_origin = np.copy(X)
    x_input = x_input.reshape(1,1,len(x_input))
    x_input_norm = preprocess_data(x_input,scaler_path=scaler_path)
    # filter X and y based on y_threshold
    above_threshold_indices = np.where(y >= y_low)
    X_filtered = X_norm[above_threshold_indices]
    x_origin_filtered = x_origin[above_threshold_indices]
    y_filtered = y[above_threshold_indices]

    below_threshold_indices = np.where(y_filtered <= y_upper)
    X_filtered = X_filtered[below_threshold_indices]
    y_filtered = y_filtered[below_threshold_indices]
    x_origin_filtered = x_origin_filtered[below_threshold_indices]

    # filter by temperature
    for idx in fix_idxes:
        feature = x_input_norm [0,0,idx]
        feature_filtered_indices = np.where(X_filtered[:,idx]==feature)
        X_filtered = X_filtered[feature_filtered_indices]
        y_filtered = y_filtered[feature_filtered_indices]
        x_origin_filtered = x_origin_filtered[feature_filtered_indices]
    
    # find the closest datapoint in the first 'time_steps' steps
    distances = np.array([distance.euclidean(x_input_norm.flatten(), x_i_norm.flatten()) for x_i_norm in X_filtered])
    closest_index = np.argmin(distances)
    closest_y = y_filtered[closest_index]
    closest_values_original = x_origin_filtered[closest_index]
    directions = []
    # determine the direction of each feature 
    x_input = x_input.reshape(K,)
    closest_values_original = x_origin_filtered[closest_index].reshape(K,)
    for i in range(K):
        if x_input[i] < closest_values_original[i]:
            directions.append("增加")
        elif x_input[i] == closest_values_original[i]:
            directions.append("保持")
        else:
            directions.append("减少")

    return closest_values_original, closest_y, directions



if __name__ == '__main__':
    X = np.load('data/X.npy')
    y = np.load("data/y.npy")
    x_input = np.array([115 ,  51.8 ,   3.75,  19.  ,  35., 39.9])
    scaler_path = "scaler.pkl"

    closest_values, closest_y, directions = recommend_adjustments(X, y, x_input, y_low =8.72, y_upper=9.25, scaler_path=scaler_path, fix_idxes=[0,3])
    print (closest_values)
    print (closest_y)
    print (directions)

