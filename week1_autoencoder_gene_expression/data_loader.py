import os
import pandas as pd
import zipfile
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def download_lincs_sample(url=None):
    """
    Downloads a small subset of LINCS L1000 data for testing.
    """
    if url is None:
        # Example CSV subset (change if you want full GSE92742!)
        url = "https://github.com/vasantm/public-datasets/raw/main/lincs_small_expression.csv.zip"

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download dataset.")

    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("data/")
    
    df = pd.read_csv("data/lincs_small_expression.csv", index_col=0)
    return df


def preprocess_data(df, test_size=0.2, batch_size=128):
    """
    Normalize gene expression and create PyTorch Dataloaders.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    X_train, X_test = train_test_split(X_scaled, test_size=test_size, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, X_test), batch_size=batch_size)

    return train_loader, test_loader, df.shape[1]

