import pandas as pd
import yaml
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_static_data(df):
    for col in df.columns:
        df[col] = df[col].astype('category').cat.codes
    df = df.drop(columns=['subject_id'], axis=1)
    return df


def preprocess_temporal_data(df):
    """
    Preprocess the input DataFrame to get the required tensors.
    """
    patient_data = []
    seq_len = 48  # Desired sequence length

    grouped = df.groupby(['subject_id', 'hadm_id'])

    for _, group in grouped:
        group = group.sort_values(by='hour')
        # Select all columns except 'subject_id', 'hadm_id', 'stay_id'
        values = torch.tensor(group.drop(
            columns=['subject_id', 'hadm_id', 'stay_id', 'hour']).values).float()

        # Ensure the sequence length <= 48
        if values.size(0) > seq_len:
            values = values[:seq_len]

        values = values.unsqueeze(0)  # Add batch dimension
        patient_data.append(values)

    return patient_data


def load_config(filepath):
    """
    Load configuration from a YAML file.
    """
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)


def load_patient_embeddings(filepath):
    """
    Load patient embeddings from a file.

    Parameters:
    filepath (str): Path to the file containing patient embeddings.

    Returns:
    torch.Tensor: Tensor containing patient embeddings.
    """
    embeddings = torch.load(filepath)
    return embeddings


def generate_masks(labels, train_ratio=0.7, val_ratio=0.1):
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, train_size=train_ratio, stratify=labels)
    val_indices, test_indices = train_test_split(
        test_indices, test_size=val_ratio/(1-train_ratio), stratify=labels[test_indices])

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask
