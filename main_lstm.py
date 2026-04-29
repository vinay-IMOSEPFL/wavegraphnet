import argparse
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.fft
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import parse_damage_label, DAMAGE_LABELS
from utils.checkpointer import save_checkpoint, checkpoint_path
from models.lstm import LSTM_baseline
from utils.logger import log_result
import pickle

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LstmDataset(Dataset):
    def __init__(self, data_map, sample_id_list, fixed_fft_bin_indices,
                 amp_means, amp_stds, lookback_fft=500):
        self.data_map              = data_map
        self.sample_id_list        = sample_id_list
        self.lookback_fft          = lookback_fft
        self.fixed_fft_bin_indices = fixed_fft_bin_indices
        self.amp_means             = amp_means
        self.amp_stds              = amp_stds
        self.num_attention_freqs   = len(fixed_fft_bin_indices)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        sig       = self.data_map[sample_id]
        num_pairs = sig.shape[1]

        fft_complex = scipy.fft.rfft(sig[: self.lookback_fft, :],
                                     n=self.lookback_fft, axis=0)
        fft_complex = fft_complex[self.fixed_fft_bin_indices, :]

        amps   = np.abs(fft_complex)
        phases = np.angle(fft_complex)
        normalized_amps = (amps - self.amp_means.reshape(1, num_pairs)) / \
                          self.amp_stds.reshape(1, num_pairs)

        x = torch.zeros((num_pairs, self.num_attention_freqs, 2), dtype=torch.float32)
        x[:, :, 0] = torch.from_numpy(normalized_amps.T).float()
        x[:, :, 1] = torch.from_numpy(phases.T).float()

        d = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if d != "undamaged" and d in DAMAGE_LABELS:
            xd, yd = float(DAMAGE_LABELS[d][0]), float(DAMAGE_LABELS[d][1])
        return x, torch.tensor([xd, yd], dtype=torch.float)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",           type=str,   default="A", choices=["A", "B"])
    parser.add_argument("--epochs",          type=int,   default=150)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--lstm_hidden_dim", type=int,   default=128,
                        help="LSTM hidden dimension (paper: 256)")
    parser.add_argument("--num_lstm_layers", type=int,   default=2,
                        help="Number of LSTM layers (paper: 3)")
    parser.add_argument("--dropout",         type=float, default=0.2,
                        help="Dropout rate (paper: 0.3)")
    parser.add_argument("--num_fft_bins",    type=int,   default=251,
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
    amp_means = np.zeros(num_sensor_pairs)
    amp_stds  = np.ones(num_sensor_pairs)

    train_loader = DataLoader(
        LstmDataset(data_map, train_ids, fixed_fft_bin_indices,
                    amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        LstmDataset(data_map, test_ids, fixed_fft_bin_indices,
                    amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTM_baseline(
        num_freqs=args.num_fft_bins,
        feature_dim_per_freq=2,
        num_sensor_pairs=num_sensor_pairs,
        lstm_hidden_dim=args.lstm_hidden_dim,
        num_lstm_layers=args.num_lstm_layers,
        decoder_hidden_dim=args.lstm_hidden_dim,
        output_dim=2,
        dropout_rate=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20)
    criterion = nn.MSELoss()

    print(f"--- LSTM | split={args.split} seed={args.seed} "
          f"lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
          f"hidden={args.lstm_hidden_dim} layers={args.num_lstm_layers} "
          f"dropout={args.dropout} fft_bins={args.num_fft_bins} ---")

    test_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}",
                    leave=False, disable=_TQDM_DISABLE)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
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
        path=checkpoint_path(args.split, "LSTM", args.seed),
        config=vars(args),
        test_loss=test_loss,
        model=model.state_dict(),
    )
    log_result(args.split, f"LSTM (seed={args.seed})", test_loss)


if __name__ == "__main__":
    main()