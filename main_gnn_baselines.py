# main_gnn_baselines.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import StandardGraphDataset, get_k_graph_edge_index
from models.gnn_baselines import FlexibleGNN
import pickle


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="A", choices=["A", "B"])
    # FIXED: Changed "mlp" to "simple_mlp" to match what FlexibleGNN expects
    parser.add_argument(
        "--model", type=str, default="simple_mlp", choices=["simple_mlp", "attention"]
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load preprocessed data and split IDs
    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    # Mock parameters for dataset initialization - replace with actual saved tensor files
    num_nodes = 12
    static_edge_index = get_k_graph_edge_index(num_nodes, self_loops=False)
    edge_feature_col_idxs = np.arange(66)  # Assuming 66 unique pairs
    fixed_fft_bin_indices = np.arange(251)
    amp_means = np.zeros(66)
    amp_stds = np.ones(66)

    # Initialize Datasets and Loaders
    train_dataset = StandardGraphDataset(
        data_map,
        train_ids,
        static_edge_index,
        edge_feature_col_idxs,
        fixed_fft_bin_indices,
        amp_means,
        amp_stds,
    )
    test_dataset = StandardGraphDataset(
        data_map,
        test_ids,
        static_edge_index,
        edge_feature_col_idxs,
        fixed_fft_bin_indices,
        amp_means,
        amp_stds,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model
    # raw_edge_feat_dim = 3 (spatial) + 251 (freqs) * 2 (amp/phase) = 505
    model = FlexibleGNN(
        encoder_type=args.model,
        processor_type="mlp",
        raw_node_feat_dim=2,
        raw_edge_feat_dim=505,
        num_attention_freqs=251,
        hidden_dim=128,
        num_gnn_proc_layers=3,
        gat_attention_heads=4,
        decoder_mlp_hidden_dim=128,
        final_output_dim=2,
        decoder_pooling_type="mean",
        num_decoder_mlp_layers=3,
        decoder_dropout_rate=0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"--- Training GNN Baseline ({args.model}) on Split {args.split} ---")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        # Wrapped the loader in tqdm to monitor speed
        loader_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False
        )

        for batch in loader_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            loader_pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
            )


if __name__ == "__main__":
    main()
