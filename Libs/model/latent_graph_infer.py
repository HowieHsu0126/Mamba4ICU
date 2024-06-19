import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
import torch_geometric.transforms as T


class GraphLearner(nn.Module):
    def __init__(self, input_dim, threshold=0.5, batch_size=None):
        """
        Initialize the GraphLearner.

        Parameters:
        input_dim (int): Dimension of input features.
        threshold (float): Threshold value for sparsifying the graph.
        batch_size (int, optional): Batch size for processing. Defaults to None.
        """
        super(GraphLearner, self).__init__()
        self.threshold_value = threshold
        self.batch_size = batch_size

        self.p = nn.Linear(input_dim, input_dim)
        self.threshold = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward pass to compute the adjacency matrix using weighted cosine similarity.

        Parameters:
        x (Tensor): Input feature matrix.

        Returns:
        Tensor: Adjacency matrix.
        """
        return self._weighted_cosine(x)

    def _weighted_cosine(self, x):
        """
        Compute the weighted cosine similarity adjacency matrix.

        Parameters:
        x (Tensor): Input feature matrix.

        Returns:
        Tensor: Adjacency matrix.
        """
        x_projected = self.p(x)
        x_normalized = F.normalize(x_projected, dim=-1)
        cosine_similarity = torch.matmul(x_normalized, x_normalized.T)
        cosine_similarity = (cosine_similarity + 1) / 2  # Scale to [0, 1]
        adjacency_matrix = self._sparsify_graph(cosine_similarity)
        return adjacency_matrix

    def _sparsify_graph(self, similarity_matrix):
        """
        Sparsify the graph based on the threshold value.

        Parameters:
        similarity_matrix (Tensor): Cosine similarity matrix.

        Returns:
        Tensor: Sparsified adjacency matrix.
        """
        mask = (similarity_matrix > self.threshold_value).float()
        sparse_matrix = similarity_matrix * mask
        sparse_matrix = (sparse_matrix - self.threshold_value) / \
            (1 - self.threshold_value)
        return self._apply_threshold(sparse_matrix * mask)

    def _apply_threshold(self, matrix):
        """
        Apply the threshold to the adjacency matrix.

        Parameters:
        matrix (Tensor): Adjacency matrix.

        Returns:
        Tensor: Thresholded adjacency matrix.
        """
        matrix[matrix < self.threshold_value] = 0
        return matrix


class LatentGraphInferencer(nn.Module):
    def __init__(self, input_dim, graph_threshold=0.5, batch_size=None, device=None, label_path='labels.csv'):
        """
        Initialize the LatentGraphInferencer.

        Parameters:
        input_dim (int): Dimension of input features.
        graph_threshold (float): Threshold value for sparsifying the graph.
        batch_size (int, optional): Batch size for processing. Defaults to None.
        device (torch.device, optional): Device to run the computations. Defaults to CPU.
        label_path (str): Path to the CSV file containing labels.
        """
        super(LatentGraphInferencer, self).__init__()
        self.device = device if device else torch.device('cpu')
        self.graph_learn = GraphLearner(
            input_dim, threshold=graph_threshold, batch_size=batch_size).to(self.device)
        self.label_path = label_path
        self.labels = self._load_labels()
        # self.transform = T.Compose([
        #     T.ToUndirected(),
        #     T.AddSelfLoops(),
        #     T.AddRandomWalkPE(walk_length=20, attr_name=None),
        # ])

    def forward(self, patient_embeddings):
        """
        Forward pass to compute the latent graph from patient embeddings.

        Parameters:
        patient_embeddings (Tensor): Patient embedding matrix.

        Returns:
        Data: PyG Data object containing the graph.
        """
        adjacency_matrix = self.graph_learn(patient_embeddings).to(self.device)
        edge_index, edge_attr = self._to_pyg_format(adjacency_matrix)
        data = Data(x=patient_embeddings, edge_index=edge_index,
                    edge_attr=edge_attr, y=self.labels.to(self.device))
        transform = T.Compose([
            T.ToUndirected(),
        ])
        return transform(data)

    def _to_pyg_format(self, adjacency_matrix):
        """
        Convert adjacency matrix to PyG format.

        Parameters:
        adjacency_matrix (Tensor): Adjacency matrix.

        Returns:
        (Tensor, Tensor): edge_index and edge_attr for PyG Data object.
        """
        row, col = adjacency_matrix.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)
        edge_attr = adjacency_matrix[row, col]
        return edge_index, edge_attr

    def _load_labels(self):
        """
        Load labels from the CSV file.

        Returns:
        Tensor: Tensor containing the labels.
        """
        labels_df = pd.read_csv(self.label_path)
        labels = torch.tensor(labels_df['is_deceased_icu'].values)
        return labels
