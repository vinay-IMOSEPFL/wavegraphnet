import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import CoupledModelDataset, get_k_graph_edge_index
from models.wavegraphnet import GNN_inv_HierarchicalAttention, DirectPathAttenuationGNN
from utils.logger import log_result
import pickle


def train_model(
    inv_model,
    fwd_model,
    train_loader,
    optimizer,
    mode="coupled",
    alpha=0.5,
    device="cpu",
):
    inv_model.train()
    if mode == "coupled":
        fwd_model.train()

    total_loss, total_loc_loss, total_fwd_loss = 0, 0, 0
    criterion = nn.MSELoss()

    loader_pbar = tqdm(train_loader, leave=False)

    for batch in loader_pbar:
        optimizer.zero_grad()

        data_inv = batch["data_inv"].to(device)
        y_true = batch["y_true"].to(device).squeeze(1)

        # 1. Inverse Pass (Always happens)
        pred_coords = inv_model(data_inv)
        loss_loc = criterion(pred_coords, y_true)

        # 2. Forward Pass (Only in coupled mode)
        if mode == "coupled":
            delta_e_true = batch["delta_e_true"].to(device)
            pred_delta_e = fwd_model(data_inv, pred_coords)
            loss_fwd = criterion(pred_delta_e, delta_e_true)

            # Coupled Loss
            loss = alpha * loss_loc + (1 - alpha) * loss_fwd

            total_fwd_loss += loss_fwd.item() * data_inv.num_graphs
            fwd_loss_val = loss_fwd.item()
        else:
            # Inverse Only Loss
            loss = loss_loc
            fwd_loss_val = 0.0

        loss.backward()

        # --- CRITICAL FIX: GRADIENT CLIPPING ---
        # Prevents exploding gradients from NaN geometric calculations
        # or extreme coordinate guesses from destroying network weights.
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), max_norm=1.0)
        if mode == "coupled":
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), max_norm=1.0)
        # ---------------------------------------

        optimizer.step()

        total_loss += loss.item() * data_inv.num_graphs
        total_loc_loss += loss_loc.item() * data_inv.num_graphs

        if mode == "coupled":
            loader_pbar.set_postfix(loc_loss=loss_loc.item(), fwd_loss=fwd_loss_val)
        else:
            loader_pbar.set_postfix(loc_loss=loss_loc.item())

    return (
        total_loss / len(train_loader.dataset),
        total_loc_loss / len(train_loader.dataset),
        total_fwd_loss / len(train_loader.dataset) if mode == "coupled" else 0.0,
    )


def evaluate(inv_model, loader, device):
    """Evaluates the inverse model's localization accuracy."""
    inv_model.eval()
    total_loc_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            data_inv = batch["data_inv"].to(device)
            y_true = batch["y_true"].to(device).squeeze(1)
            pred_coords = inv_model(data_inv)
            loss_loc = criterion(pred_coords, y_true)
            total_loc_loss += loss_loc.item() * data_inv.num_graphs
    return total_loc_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="A", choices=["A", "B"])
    parser.add_argument(
        "--mode", type=str, default="coupled", choices=["coupled", "inverse_only"]
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    num_nodes = 12
    num_sensor_pairs = list(data_map.values())[0].shape[1]

    inv_static_edge_index = get_k_graph_edge_index(num_nodes, self_loops=False)
    inv_edge_feature_col_idxs = np.arange(num_sensor_pairs)
    fwd_propagation_col_idxs = np.arange(num_sensor_pairs)
    fixed_fft_bin_indices = np.arange(251)

    amp_means = np.zeros(num_sensor_pairs)
    amp_stds = np.ones(num_sensor_pairs)
    average_baseline_energy_profile = torch.zeros(num_sensor_pairs)

    dataset_params = {
        "data_map": data_map,
        "inv_static_edge_index": inv_static_edge_index,
        "inv_edge_feature_col_idxs": inv_edge_feature_col_idxs,
        "fwd_propagation_col_idxs": fwd_propagation_col_idxs,
        "fixed_fft_bin_indices": fixed_fft_bin_indices,
        "amp_means": amp_means,
        "amp_stds": amp_stds,
        "lookback_fft": 500,
        "average_baseline_energy_profile": average_baseline_energy_profile,
        "global_max_delta_e": 1.0,
    }

    train_dataset = CoupledModelDataset(sample_id_list=train_ids, **dataset_params)
    test_dataset = CoupledModelDataset(sample_id_list=test_ids, **dataset_params)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    inv_model = GNN_inv_HierarchicalAttention(
        hidden_dim=128,
        raw_node_feat_dim=2,
        num_attention_freqs=251,
        num_gnn_proc_layers=3,
        gat_attention_heads=4,
        decoder_mlp_hidden_dim=128,
        final_output_dim=2,
        decoder_pooling_type="mean",
    ).to(device)

    fwd_model = DirectPathAttenuationGNN(
        raw_node_feat_dim=2,
        physical_edge_feat_dim=6,
        hidden_dim=128,
        num_propagation_pairs=num_sensor_pairs,
        num_interaction_layers=4,
    ).to(device)

    # Only optimize the forward model parameters if we are in coupled mode
    if args.mode == "coupled":
        optimizer = optim.Adam(
            list(inv_model.parameters()) + list(fwd_model.parameters()), lr=args.lr
        )
    else:
        optimizer = optim.Adam(inv_model.parameters(), lr=args.lr)

    mode_display = "Coupled" if args.mode == "coupled" else "Inverse Only"
    print(f"--- Training WaveGraphNet ({mode_display}) on Split {args.split} ---")
    test_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        loss, loc_loss, fwd_loss = train_model(
            inv_model, fwd_model, train_loader, optimizer, mode=args.mode, device=device
        )

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            test_loss = evaluate(inv_model, test_loader, device)
            if args.mode == "coupled":
                print(
                    f"Epoch {epoch:03d} | Train Loc: {loc_loss:.4f} | Train Fwd: {fwd_loss:.4f} | Test Loc: {test_loss:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | Train Loc: {loc_loss:.4f} | Test Loc: {test_loss:.4f}"
                )

    log_result(args.split, f"WaveGraphNet ({mode_display})", test_loss)


if __name__ == "__main__":
    main()
