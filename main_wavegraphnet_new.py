import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import CoupledModelDataset, get_k_graph_edge_index
from models.wavegraphnet_new import InverseGNN, ForwardPhysicsGNN, DynamicWeightedLoss
from utils.logger import log_result
import pickle


def train_model(
    inv_model,
    fwd_model,
    dynamic_loss,
    train_loader,
    optimizer,
    mode="coupled",
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

        # 1. Inverse Pass: Predict Damage Location
        pred_coords = inv_model(data_inv)
        loss_loc = criterion(pred_coords, y_true)

        if mode == "coupled":
            # 2. Forward Pass: Predict Energy Deviation
            # FIX: fwd_model now returns [batch, num_propagation_pairs] which
            # matches delta_e_true shape [batch, num_propagation_pairs].
            delta_e_true = batch["delta_e_true"].to(device)
            pred_delta_e = fwd_model(data_inv, pred_coords)
            loss_fwd = criterion(pred_delta_e, delta_e_true)

            # Use the Learnable Dynamic Loss Weighting
            loss = dynamic_loss([loss_loc, loss_fwd])

            total_fwd_loss += loss_fwd.item() * data_inv.num_graphs
            fwd_loss_val = loss_fwd.item()
        else:
            loss = loss_loc
            fwd_loss_val = 0.0

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), max_norm=1.0)
        if mode == "coupled":
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * data_inv.num_graphs
        total_loc_loss += loss_loc.item() * data_inv.num_graphs
        loader_pbar.set_postfix(loc=loss_loc.item(), fwd=fwd_loss_val)

    return (
        total_loss / len(train_loader.dataset),
        total_loc_loss / len(train_loader.dataset),
        total_fwd_loss / len(train_loader.dataset) if mode == "coupled" else 0.0,
    )


def evaluate(inv_model, loader, device):
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
    parser.add_argument("--split", type=str, default="B", choices=["A", "B"])
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

    num_sensor_pairs = list(data_map.values())[0].shape[1]
    inv_static_edge_index = get_k_graph_edge_index(12, self_loops=False)

    dataset_params = {
        "data_map": data_map,
        "inv_static_edge_index": inv_static_edge_index,
        "inv_edge_feature_col_idxs": np.arange(num_sensor_pairs),
        "fwd_propagation_col_idxs": np.arange(num_sensor_pairs),
        "fixed_fft_bin_indices": np.arange(251),
        "amp_means": np.zeros(num_sensor_pairs),
        "amp_stds": np.ones(num_sensor_pairs),
        "lookback_fft": 500,
        "average_baseline_energy_profile": torch.zeros(num_sensor_pairs),
        "global_max_delta_e": 1.0,
    }

    train_loader = DataLoader(
        CoupledModelDataset(sample_id_list=train_ids, **dataset_params),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        CoupledModelDataset(sample_id_list=test_ids, **dataset_params),
        batch_size=args.batch_size,
        shuffle=False,
    )

    inv_model = InverseGNN(node_in=2, edge_in=505, hidden=128).to(device)

    # FIX: pass num_propagation_pairs so ForwardPhysicsGNN can reshape its
    # [batch * 132] output to [batch, 66] matching delta_e_true's shape.
    fwd_model = ForwardPhysicsGNN(
        edge_in=6, hidden=128, num_propagation_pairs=num_sensor_pairs
    ).to(device)

    dynamic_loss = DynamicWeightedLoss(num_losses=2).to(device)

    optimizer = optim.Adam(
        list(inv_model.parameters())
        + list(fwd_model.parameters())
        + list(dynamic_loss.parameters()),
        lr=args.lr,
    )

    print(f"--- Training Improved WaveGraphNet on Split {args.split} ---")
    test_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        loss, loc_loss, fwd_loss = train_model(
            inv_model,
            fwd_model,
            dynamic_loss,
            train_loader,
            optimizer,
            mode=args.mode,
            device=device,
        )

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            test_loss = evaluate(inv_model, test_loader, device)
            print(
                f"Epoch {epoch:03d} | Train Loc: {loc_loss:.4f} | Test Loc: {test_loss:.4f}"
            )

    log_result(args.split, "WaveGraphNet (Improved)", test_loss)


if __name__ == "__main__":
    main()
