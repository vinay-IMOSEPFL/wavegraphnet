import argparse
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import StandardGraphDataset, get_k_graph_edge_index
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.gnn_baselines import FlexibleGNN
from utils.logger import log_result
import pickle

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            total += criterion(model(batch), batch.y).item() * batch.num_graphs
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",          type=str,   default="A", choices=["A", "B"])
    parser.add_argument("--model",          type=str,   default="simple_mlp",
                        choices=["simple_mlp", "attention"])
    parser.add_argument("--epochs",         type=int,   default=150)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=0.001)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--hidden_dim",     type=int,   default=128,
                        help="Embedding dimension (paper: 256)")
    parser.add_argument("--num_gnn_layers", type=int,   default=3,
                        help="Message-passing layers (paper: 4)")
    parser.add_argument("--gat_heads",      type=int,   default=4,
                        help="GAT attention heads for attention encoder (paper: 16)")
    parser.add_argument("--num_fft_bins",   type=int,   default=251,
                        help="FFT bins; paper=256")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lookback_fft          = (args.num_fft_bins - 1) * 2
    fixed_fft_bin_indices = np.arange(args.num_fft_bins)
    raw_edge_feat_dim     = 3 + args.num_fft_bins * 2

    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    num_sensor_pairs      = list(data_map.values())[0].shape[1]
    static_edge_index     = get_k_graph_edge_index(12, self_loops=False)
    edge_feature_col_idxs = np.arange(num_sensor_pairs)
    amp_means = np.zeros(num_sensor_pairs)
    amp_stds  = np.ones(num_sensor_pairs)

    train_loader = DataLoader(
        StandardGraphDataset(data_map, train_ids, static_edge_index,
                             edge_feature_col_idxs, fixed_fft_bin_indices,
                             amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        StandardGraphDataset(data_map, test_ids, static_edge_index,
                             edge_feature_col_idxs, fixed_fft_bin_indices,
                             amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=False)

    model = FlexibleGNN(
        encoder_type=args.model,
        processor_type="mlp",
        raw_node_feat_dim=2,
        raw_edge_feat_dim=raw_edge_feat_dim,
        num_attention_freqs=args.num_fft_bins,
        hidden_dim=args.hidden_dim,
        num_gnn_proc_layers=args.num_gnn_layers,
        gat_attention_heads=args.gat_heads,
        decoder_mlp_hidden_dim=args.hidden_dim,
        final_output_dim=2,
        decoder_pooling_type="mean",
        num_decoder_mlp_layers=3,
        decoder_dropout_rate=0.2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20)
    criterion = nn.MSELoss()

    label = f"GNN ({args.model})"
    print(f"--- {label} | split={args.split} seed={args.seed} "
          f"lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
          f"hidden={args.hidden_dim} layers={args.num_gnn_layers} "
          f"gat_heads={args.gat_heads} fft_bins={args.num_fft_bins} ---")

    test_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}",
                    leave=False, disable=_TQDM_DISABLE)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            if not _TQDM_DISABLE:
                pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)
        scheduler.step(train_loss)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch:03d} | Train: {train_loss:.6f} | "
                  f"Test: {test_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------ #
    # Save checkpoint                                                       #
    # ------------------------------------------------------------------ #
    save_checkpoint(
        path=checkpoint_path(args.split, label, args.seed),
        config=vars(args),
        test_loss=test_loss,
        model=model.state_dict(),
    )
    log_result(args.split, f"{label} (seed={args.seed})", test_loss)


if __name__ == "__main__":
    main()