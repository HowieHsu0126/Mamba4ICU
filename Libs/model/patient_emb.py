from enum import Enum
import torch
import torch.nn as nn
from Libs.utils.utils import load_data, preprocess_temporal_data, preprocess_static_data


class ActivationType(Enum):
    SIN = "sin"
    COS = "cos"


class PeriodicActivation(nn.Module):
    def __init__(self, time_features, activation: ActivationType, device=None):
        """
        Initialize the PeriodicActivation module.
        """
        super(PeriodicActivation, self).__init__()
        self.device = device
        self.activation_function = torch.sin if activation == ActivationType.SIN else torch.cos
        self.periodic_weights = nn.Parameter(
            torch.randn(1, time_features - 1, device=device))
        self.periodic_biases = nn.Parameter(
            torch.randn(time_features - 1, device=device))
        self.linear_weight = nn.Parameter(torch.randn(1, 1, device=device))
        self.linear_bias = nn.Parameter(torch.randn(1, device=device))

        # Improved weight initialization
        nn.init.xavier_uniform_(self.periodic_weights)
        nn.init.xavier_uniform_(self.linear_weight)

    def forward(self, timestamps):
        """
        Forward pass for PeriodicActivation.
        """
        periodic_component = self.activation_function(
            torch.matmul(timestamps, self.periodic_weights) +
            self.periodic_biases
        )
        linear_component = torch.matmul(
            timestamps, self.linear_weight) + self.linear_bias
        return torch.cat([periodic_component, linear_component], -1)


class TimeEmbedding(nn.Module):
    def __init__(self, time_features, activation: ActivationType, device=None):
        """
        Initialize the TimeEmbedding module.
        """
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.time_features = time_features
        self.activation = activation
        self.periodic_activations = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

    def build_layers(self, num_variables):
        """
        Build the periodic activation and fully connected layers.
        """
        self.periodic_activations = nn.ModuleList(
            [PeriodicActivation(self.time_features, self.activation, self.device).to(
                self.device) for _ in range(num_variables)]
        )
        self.fc_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(self.time_features, self.time_features).to(
                    self.device),
                nn.LayerNorm(self.time_features).to(
                    self.device),  # Layer Normalization
                nn.ReLU(),  # Activation Function
                nn.Dropout(p=0.5)  # Dropout for regularization
            ) for _ in range(num_variables)]
        )

    def forward(self, x):
        """
        Forward pass for TimeEmbedding.
        """
        batch_size, seq_len, num_features = x.size()

        if len(self.periodic_activations) == 0 or len(self.fc_layers) == 0:
            self.build_layers(num_features)

        outputs = []
        for i in range(num_features):
            feature = x[:, :, i:i + 1]  # Select the i-th variable
            periodic_output = self.periodic_activations[i](
                feature.view(batch_size * seq_len, 1).to(self.device))
            output = self.fc_layers[i](periodic_output.view(
                batch_size * seq_len, -1))  # Reshape for LayerNorm
            output = output.view(batch_size, seq_len, -1)
            outputs.append(output)

        # Stack outputs along the last dimension and then aggregate
        stacked_outputs = torch.stack(outputs, dim=-1)
        aggregated_output = torch.mean(stacked_outputs, dim=-1)
        return aggregated_output

    @staticmethod
    def prepare_data(file_path):
        """
        Load and preprocess the temporal data.
        """
        df = load_data(file_path)
        preprocessed_data = preprocess_temporal_data(df)
        return preprocessed_data


class StaticEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_unique_values, device=None):
        """
        Initialize the StaticEmbedding module.
        """
        super(StaticEmbedding, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_unique_values = num_unique_values
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(unique_count, embedding_dim).to(self.device)
            for col, unique_count in num_unique_values.items()
        })

    def forward(self, x):
        """
        Forward pass for StaticEmbedding.
        """
        embedded_cols = [self.embeddings[col](x[:, i].to(self.device))
                         for i, col in enumerate(self.embeddings.keys())]
        concatenated = torch.cat(embedded_cols, dim=1)
        return concatenated

    @staticmethod
    def prepare_data(file_path):
        """
        Load and preprocess the static data.
        """
        df = load_data(file_path)
        preprocessed_data = preprocess_static_data(df)
        num_unique_values = {col: preprocessed_data[col].nunique()
                             for col in preprocessed_data.columns}
        return preprocessed_data, num_unique_values


class PatientEmbedding(nn.Module):
    def __init__(self, time_embedding: TimeEmbedding, static_embedding: StaticEmbedding, final_embedding_dim, device=None):
        """
        Initialize the PatientEmbedding module.
        """
        super(PatientEmbedding, self).__init__()
        self.device = device
        self.time_embedding = time_embedding.to(self.device)
        self.static_embedding = static_embedding.to(self.device)
        self.final_fc = nn.Sequential(
            nn.Linear(
                time_embedding.time_features + static_embedding.embedding_dim *
                len(static_embedding.num_unique_values),
                final_embedding_dim
            ).to(self.device),
            nn.LayerNorm(final_embedding_dim).to(
                self.device),  # Layer Normalization
            nn.ReLU(),  # Activation Function
            nn.Dropout(p=0.5)  # Dropout for regularization
        )

    def forward(self, time_data, static_data):
        """
        Forward pass for PatientEmbedding.
        """
        if static_data.size(0) == 0:
            raise ValueError("Empty tensor detected in static_data")

        time_data = time_data.to(self.device)
        static_data = static_data.to(self.device)

        time_embedded = self.time_embedding(time_data)
        static_embedded = self.static_embedding(static_data)

        if time_embedded.size(0) == 0 or static_embedded.size(0) == 0:
            raise ValueError(
                "Empty tensor detected in time_embedded or static_embedded")

        combined_embedded = torch.cat(
            (time_embedded, static_embedded.unsqueeze(1).expand(-1, time_embedded.size(1), -1)), dim=2
        )
        combined_embedded = combined_embedded.view(combined_embedded.size(
            0) * combined_embedded.size(1), -1)  # Reshape for LayerNorm
        final_embedding = self.final_fc(combined_embedded)
        final_embedding = final_embedding.view(time_embedded.size(
            0), time_embedded.size(1), -1)  # Reshape back to original dimensions

        # Take the mean across the sequence length dimension to get a fixed-size embedding
        final_embedding = final_embedding.mean(dim=1)
        return final_embedding
