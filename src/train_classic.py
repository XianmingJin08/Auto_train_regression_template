from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA


def preprocess_data(X,y,test_ratio=0.2):
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    N,T,K = X_train.shape
    X_train = X_train.reshape(-1,T*K)
    X_test = X_test.reshape(-1,T*K)
    X_trian_origin, X_test_origin = np.copy(X_train), np.copy(X_test)
    # Initialize the scaler and fit it on the training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Normalize the data using the scaler
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    with open("scaler_classic.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    pca = PCA(n_components=1)

    # Perform dimensionality reduction
    X_train_reduced = pca.fit_transform(X_trian_origin)
    X_test_reduced = pca.transform(X_test_origin)

    return X_train, X_test, y_train, y_test, X_trian_origin, X_test_origin, X_train_reduced, X_test_reduced


def train_random_forest(X_train, y_train):
    # Initialize and train the Random Forest regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    return rf_regressor

def test_random_forest(model, X_test, y_test):
    # Make predictions using the trained Random Forest model
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, mse, mae

def plot_mae_feature(y_test, y_pred, X_test, feature_idxes=[], idx_names=[]):
    num_features = len(feature_idxes)
    num_cols = min(3, num_features)
    num_rows = math.ceil(num_features / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
    axes = axes.reshape((num_rows,num_cols))
    mea_feature = [abs(true - pred) for true, pred in zip(y_test, y_pred)]

    for i, (idx, name) in enumerate(zip(feature_idxes, idx_names)):
        feature = X_test[:, idx]
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx] if num_features > 1 else axes
        ax.scatter(feature, mea_feature)
        ax.set_xlabel('Feature {}'.format(name))
        ax.set_ylabel('MAE')
        ax.set_title('MAE vs Feature {}'.format(name))

    for j in range(i + 1, num_rows * num_cols):
        row_idx = j // num_cols
        col_idx = j % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

def plot_y_feature(y_test, X_test, feature_idxes=[], idx_names=[]):
    num_features = len(feature_idxes)
    num_cols = min(5, num_features)
    num_rows = math.ceil(num_features / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
    axes = axes.reshape(num_rows,num_cols)
    for i, (idx, name) in enumerate(zip(feature_idxes, idx_names)):
        feature = X_test[:, idx]
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx] if num_features > 1 else axes
        ax.scatter(feature, y_test)
        ax.set_xlabel('Feature {}'.format(name))
        ax.set_ylabel('y')
        ax.set_title('y vs Feature {}'.format(name))

    for j in range(i + 1, num_rows * num_cols):
        row_idx = j // num_cols
        col_idx = j % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

X = np.load("data/X.npy")
y = np.load("data/y.npy")
print (X.shape)
# Process the dataset
X_train, X_test, y_train, y_test, X_train_origin, X_test_origin, X_train_reduced, X_test_reduced  = preprocess_data(X,y)

print (X_train_reduced.shape)
# Train the Random Forest model
rf_model = train_random_forest(X_train, y_train)

# Test the Random Forest model
y_pred, mse, mae = test_random_forest(rf_model, X_test, y_test)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

plot_mae_feature(y_test,y_pred,X_test_origin,feature_idxes=[0,1,2,3,4],idx_names=["top","bot","pressure","temp","flow_in"])

plot_y_feature(y_test,X_test_origin,feature_idxes=[0,1,2,3,4],idx_names=["top","bot","pressure","temp","flow_in"])
