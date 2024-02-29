import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    print(data.head())
    print(data.shape)
    print(data.describe())
    print(data['Outcome'].value_counts())
    print(data.groupby('Outcome').mean())

def split_features_and_target(data, target_name='Outcome'):
    X = data.drop(columns=target_name, axis=1)
    Y = data[target_name]
    return X, Y

def split_data(X, Y, test_size=0.2, random_state=2, stratify=None):
    return train_test_split(X, Y, test_size=test_size, stratify=stratify, random_state=random_state)

def standardize_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    return standardized_data, scaler
