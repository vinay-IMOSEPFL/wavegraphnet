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
from utils.data_loader import CoupledModelDataset, get_k_graph_edge_index
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.wavegraphnet import GNN_inv_HierarchicalAttention, DirectPathAttenuationGNN
from utils.logger import log_result
import pickle

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(inv_model, fwd_model, train_loader, optimizer,
                mode="coupled", alpha=0.5, device="cpu"):
    inv_model.train()
    if mode == "coupled":
        fwd_model.train()

    total_loss = total_loc = total_fwd = 0.0
    criterion  = nn.MSELoss()
    pbar       = tqdm(train_loader, leave=False, disable=_TQDM_DISABLE)

    for batch in pbar:
        optimizer.zero_grad()
        data_inv    = batch["data_inv"].to(device)
        y_true      = batch["y_true"].to(device).squeeze(1)
        pred_coords = inv_model(data_inv)
        loss_loc    = criterion(pred_coords, y_true)

        if mode == "coupled":
            delta_e_true = batch["delta_e_true"].to(device)
            pred_delta_e = fwd_model(data_inv, pred_coords)
            loss_fwd     = criterion(pred_delta_e, delta_e_true)
            loss         = alpha * loss_loc + (1 - alpha) * loss_fwd
            total_fwd   += loss_fwd.item() * data_inv.num_graphs
            fwd_val      = loss_fwd.item()
        else:
            loss    = loss_loc
            fwd_val = 0.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), max_norm=1.0)
        if mode == "coupled":
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()     * data_inv.num_graphs
        total_loc  += loss_loc.item() * data_inv.num_graphs
        if not _TQDM_DISABLE:
            pbar.set_postfix(loc=loss_loc.item(), fwd=fwd_val)

    n = len(train_loader.dataset)
    return total_loss / n, total_loc / n, total_fwd / n if mode == "coupled" else 0.0


def evaluate(inv_model, loader, device):
    inv_model.eval()
    total     = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            data_inv = batch["data_inv"].to(device)
            y_true   = batch["y_true"].to(device).squeeze(1)
            total   += criterion(inv_model(data_inv), y_true).item() * data_inv.num_graphs
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",      type=str,   default="A", choices=["A", "B"])
    parser.add_argument("--mode",       type=str,   default="coupled",
                        choices=["coupled", "inverse_only"])
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=0.001)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--inv_hidden_dim",         type=int, default=128,
                        help="Inverse-branch hidden dim (paper: 256)")
    parser.add_argument("--fwd_hidden_dim",         type=int, default=128,
                        help="Forward-branch hidden dim (paper: 512)")
    parser.add_argument("--num_interaction_layers", type=int, default=4,
                        help="Forward-branch message-passing layers (paper: 8)")
    parser.add_argument("--num_fft_bins",           type=int, default=251,
                        help="FFT bins; paper=256")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lookback_fft          = (args.num_fft_bins - 1) * 2
    fixed_fft_bin_indices = np.arange(args.num_fft_bins)

    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    num_sensor_pairs = list(data_map.values())[0].shape[1]
    inv_edge_index   = get_k_graph_edge_index(12, self_loops=False)

    dataset_params = dict(
        data_map=data_map,
        inv_static_edge_index=inv_edge_index,
        inv_edge_feature_col_idxs=np.arange(num_sensor_pairs),
        fwd_propagation_col_idxs=np.arange(num_sensor_pairs),
        fixed_fft_bin_indices=fixed_fft_bin_indices,
        amp_means=np.zeros(num_sensor_pairs),
        amp_stds=np.ones(num_sensor_pairs),
        lookback_fft=lookback_fft,
        average_baseline_energy_profile=torch.zeros(num_sensor_pairs),
        global_max_delta_e=1.0,
    )

    train_loader = DataLoader(
        CoupledModelDataset(sample_id_list=train_ids, **dataset_params),
        batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        CoupledModelDataset(sample_id_list=test_ids, **dataset_params),
        batch_size=args.batch_size, shuffle=False)

    inv_model = GNN_inv_HierarchicalAttention(
        hidden_dim=args.inv_hidden_dim,
        raw_node_feat_dim=2,
        num_attention_freqs=args.num_fft_bins,
        num_gnn_proc_layers=4,
        gat_attention_heads=4,
        decoder_mlp_hidden_dim=args.inv_hidden_dim,
        final_output_dim=2,
        decoder_pooling_type="mean",
    ).to(device)

    fwd_model = DirectPathAttenuationGNN(
        raw_node_feat_dim=2,
        physical_edge_feat_dim=6,
        hidden_dim=args.fwd_hidden_dim,
        num_propagation_pairs=num_sensor_pairs,
        num_interaction_layers=args.num_interaction_layers,
    ).to(device)

    params = list(inv_model.parameters())
    if args.mode == "coupled":
        params += list(fwd_model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20)

    mode_label = "Coupled" if args.mode == "coupled" else "Inverse Only"
    full_label = f"WaveGraphNet ({mode_label})"
    print(f"--- {full_label} | split={args.split} seed={args.seed} "
          f"lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
          f"inv_h={args.inv_hidden_dim} fwd_h={args.fwd_hidden_dim} "
          f"mp={args.num_interaction_layers} fft_bins={args.num_fft_bins} ---")

    test_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, loc_loss, fwd_loss = train_model(
            inv_model, fwd_model, train_loader, optimizer,
            mode=args.mode, device=device)
        scheduler.step(loc_loss)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            test_loss = evaluate(inv_model, test_loader, device)
            lr_now    = optimizer.param_groups[0]["lr"]
            if args.mode == "coupled":
                print(f"Epoch {epoch:03d} | Loc: {loc_loss:.4f} | "
                      f"Fwd: {fwd_loss:.4f} | Test: {test_loss:.4f} | LR: {lr_now:.2e}")
            else:
                print(f"Epoch {epoch:03d} | Loc: {loc_loss:.4f} | "
                      f"Test: {test_loss:.4f} | LR: {lr_now:.2e}")

    # ------------------------------------------------------------------ #
    # Save checkpoint — both branches so visualizations can use either     #
    # ------------------------------------------------------------------ #
    save_checkpoint(
        path=checkpoint_path(args.split, full_label, args.seed),
        config=vars(args),
        test_loss=test_loss,
        inv_model=inv_model.state_dict(),
        fwd_model=fwd_model.state_dict(),
    )
    log_result(args.split, f"{full_label} (seed={args.seed})", test_loss)


if __name__ == "__main__":
    main()