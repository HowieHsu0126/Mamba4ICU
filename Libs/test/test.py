import logging

import torch
from Libs.exp.train import test, train
from Libs.model.graph_mamba import GCN
from Libs.model.latent_graph_infer import LatentGraphInferencer
from Libs.model.patient_emb import (ActivationType, PatientEmbedding,
                                    StaticEmbedding, TimeEmbedding)
from Libs.utils.utils import (generate_masks, load_config, load_data,
                              load_patient_embeddings, preprocess_static_data,
                              preprocess_temporal_data)
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def test_embedding(config_path):
    # Set up logging
    setup_logging()

    # Load configuration from YAML file
    config = load_config(config_path)
    logging.info("Configuration loaded successfully.")

    # Extract configuration parameters
    activation = ActivationType[config['activation'].upper()]
    time_features = config['temporal_embedding_dim']
    static_embedding_dim = config['static_embedding_dim']
    final_embedding_dim = config['final_embedding_dim']

    # Initialize the embedding models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_embedding = TimeEmbedding(
        time_features, activation, device).to(device)
    logging.info(
        "TimeEmbedding model initialized and moved to device: %s", device)

    # Prepare static data and get unique value counts for each column
    preprocessed_static_data, num_unique_values = StaticEmbedding.prepare_data(
        config['static_file_path'])
    static_embedding = StaticEmbedding(
        static_embedding_dim, num_unique_values).to(device)
    logging.info(
        "StaticEmbedding model initialized and moved to device: %s", device)

    # Initialize PatientEmbedding
    patient_embedding = PatientEmbedding(
        time_embedding, static_embedding, final_embedding_dim, device)
    logging.info(
        "PatientEmbedding model initialized and moved to device: %s", device)

    # Load and preprocess time-series data from CSV
    df = load_data(config['ts_file_path'])
    patient_data = preprocess_temporal_data(df)
    logging.info("Time-series data loaded and preprocessed.")

    # Convert static dataframe to tensor
    static_tensor = torch.tensor(
        preprocessed_static_data.values, dtype=torch.long).to(device)
    logging.info(
        "Static data converted to tensor and moved to device: %s", device)

    # Compute patient embeddings
    patient_embeddings = []
    # Testing PatientEmbedding for each patient
    for i, time_values in enumerate(patient_data):
        time_values = time_values.to(device)
        static_values = static_tensor[i:i+1]

        # Add debug info for input sizes
        logging.debug(
            f"Patient {i}: time_values size: {time_values.size()}, static_values size: {static_values.size()}")

        # Check for empty static_values and skip if necessary
        if static_values.size(0) == 0:
            logging.warning(
                f"Patient {i} has empty static_values. Skipping this patient.")
            continue

        try:
            with torch.no_grad():
                output = patient_embedding(time_values, static_values)
                patient_embeddings.append(output)
                logging.info(f"Patient {i} embedding shape: {output.shape}")
        except Exception as e:
            logging.error(f"Error processing patient {i}: {e}")
            logging.error(
                f"time_values size: {time_values.size()}, static_values size: {static_values.size()}")
            raise e

        # Clear cached memory
        torch.cuda.empty_cache()

    # Stack all patient embeddings and save to file
    patient_embeddings = torch.stack(patient_embeddings).squeeze(1)
    embedding_save_path = config.get(
        'emb4test_save_path', 'patient_embeddings.pt')
    torch.save(patient_embeddings, embedding_save_path)
    logging.info(f"Patient embeddings saved to {embedding_save_path}")

    # Release GPU memory
    del patient_embedding, static_embedding, time_embedding
    torch.cuda.empty_cache()


def test_graph(config_path):
    # Set up logging
    setup_logging()

    # Load configuration from YAML file
    config = load_config(config_path)
    logging.info("Configuration loaded successfully.")

    # Extract configuration parameters
    final_embedding_dim = config['final_embedding_dim']
    graph_threshold = config.get('graph_threshold', 0.5)
    batch_size = config.get('batch_size', 64)
    label_file_path = config.get('label_file_path', 'labels.csv')

    # Initialize the embedding models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed patient embeddings
    patient_embeddings = load_patient_embeddings(
        config['emb4test_save_path']).to(device)
    logging.info(
        f"Patient embeddings loaded with shape: {patient_embeddings.shape}")

    # Initialize LatentGraphInferencer
    latent_graph_inferencer = LatentGraphInferencer(
        final_embedding_dim, graph_threshold, batch_size, device, label_file_path)
    logging.info(
        "LatentGraphInferencer model initialized and moved to device: %s", device)

    # Compute adjacency matrix
    with torch.no_grad():
        adjacency_matrix = latent_graph_inferencer(patient_embeddings)

    # Release GPU memory
    del latent_graph_inferencer, patient_embeddings
    torch.cuda.empty_cache()

    print(adjacency_matrix)
    graph4test_save_path = config.get(
        'graph4test_save_path', 'graph4test_save_path')
    torch.save(adjacency_matrix, graph4test_save_path)


def test_gnn(config_path):
    # Set up logging
    setup_logging()

    # Load configuration from YAML file
    config = load_config(config_path)
    logging.info("Configuration loaded successfully.")

    # Extract configuration parameters
    final_embedding_dim = config['final_embedding_dim']
    hidden_channels = config.get('hidden_channels', 16)
    num_classes = config.get('num_classes', 3)
    learning_rate = config.get('learning_rate', 0.01)
    weight_decay = config.get('weight_decay', 5e-4)
    epochs = config.get('epochs', 200)
    graph_threshold = config.get('graph_threshold', 0.5)
    batch_size = config.get('batch_size', 64)
    label_file_path = config.get('label_file_path', 'labels.csv')

    # Initialize the embedding models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed patient embeddings and labels
    patient_embeddings = load_patient_embeddings(
        config['emb4test_save_path']).to(device)
    logging.info(
        f"Patient embeddings loaded with shape: {patient_embeddings.shape}")

    # Initialize LatentGraphInferencer
    latent_graph_inferencer = LatentGraphInferencer(
        final_embedding_dim, graph_threshold, batch_size, device, label_file_path)
    logging.info(
        "LatentGraphInferencer model initialized and moved to device: %s", device)

    # Generate train, validation, and test masks
    labels = latent_graph_inferencer.labels.to(device)
    train_mask, val_mask, test_mask = generate_masks(labels.cpu().numpy())

    # Create PyG data object
    with torch.no_grad():
        data = latent_graph_inferencer(patient_embeddings)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(
        f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')

    # Initialize the GCN model
    model = GCN(patient_embeddings.size(
        1), hidden_channels, num_classes).to(device)
    model.device = device  # Add device attribute to the model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                  num_neighbors=[3, 2], batch_size=batch_size,
                                  subgraph_type='induced', shuffle=True)

    val_loader = NeighborLoader(data, input_nodes=data.val_mask,
                                num_neighbors=[3, 2], batch_size=batch_size,
                                subgraph_type='induced', shuffle=True)

    test_loader = NeighborLoader(data, input_nodes=data.test_mask,
                                 num_neighbors=[3, 2], batch_size=batch_size,
                                 subgraph_type='induced', shuffle=True)

    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, train_loader)

        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, train_loader), test(
                model, val_loader), test(model, test_loader)
            logging.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
