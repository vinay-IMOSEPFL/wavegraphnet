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
from models.wavegraphnet_new import InverseGNN, ForwardPhysicsGNN, DynamicWeightedLoss
from utils.logger import log_result
import pickle

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(inv_model, fwd_model, dynamic_loss, train_loader,
                optimizer, mode="coupled", device="cpu"):
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
            loss         = dynamic_loss([loss_loc, loss_fwd])
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
    parser.add_argument("--split",        type=str,   default="B", choices=["A", "B"])
    parser.add_argument("--mode",         type=str,   default="coupled",
                        choices=["coupled", "inverse_only"])
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_fft_bins", type=int,   default=251,
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

    inv_edge_feat_dim = 3 + args.num_fft_bins * 2

    inv_model    = InverseGNN(node_in=2, edge_in=inv_edge_feat_dim, hidden=128).to(device)
    fwd_model    = ForwardPhysicsGNN(edge_in=6, hidden=128,
                                     num_propagation_pairs=num_sensor_pairs).to(device)
    dynamic_loss = DynamicWeightedLoss(num_losses=2).to(device)

    optimizer = optim.Adam(
        list(inv_model.parameters()) +
        list(fwd_model.parameters()) +
        list(dynamic_loss.parameters()),
        lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20)

    print(f"--- WaveGraphNet (Improved) | split={args.split} seed={args.seed} "
          f"lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
          f"fft_bins={args.num_fft_bins} ---")

    test_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, loc_loss, fwd_loss = train_model(
            inv_model, fwd_model, dynamic_loss, train_loader,
            optimizer, mode=args.mode, device=device)
        scheduler.step(loc_loss)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            test_loss = evaluate(inv_model, test_loader, device)
            print(f"Epoch {epoch:03d} | Loc: {loc_loss:.4f} | "
                  f"Test: {test_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ------------------------------------------------------------------ #
    # Save checkpoint — all three components for full reproducibility      #
    # ------------------------------------------------------------------ #
    save_checkpoint(
        path=checkpoint_path(args.split, "WaveGraphNet (Improved)", args.seed),
        config=vars(args),
        test_loss=test_loss,
        inv_model=inv_model.state_dict(),
        fwd_model=fwd_model.state_dict(),
        dynamic_loss=dynamic_loss.state_dict(),
    )
    log_result(args.split, f"WaveGraphNet (Improved) (seed={args.seed})", test_loss)


if __name__ == "__main__":
    main()