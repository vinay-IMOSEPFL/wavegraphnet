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
from models.cnn1d import PaperCnnBaseline
from utils.logger import log_result
import pickle

_TQDM_DISABLE = not sys.stdout.isatty()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Cnn1DDataset(Dataset):
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

        x = torch.zeros((num_pairs * 2, self.num_attention_freqs), dtype=torch.float32)
        for p in range(num_pairs):
            x[p * 2,     :] = torch.from_numpy(normalized_amps[:, p]).float()
            x[p * 2 + 1, :] = torch.from_numpy(phases[:, p]).float()

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
    parser.add_argument("--split",        type=str,   default="A", choices=["A", "B"])
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--num_fft_bins", type=int,   default=251,
                        help="FFT bins; paper=256 (sets lookback_fft=(bins-1)*2)")
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
        Cnn1DDataset(data_map, train_ids, fixed_fft_bin_indices,
                     amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        Cnn1DDataset(data_map, test_ids, fixed_fft_bin_indices,
                     amp_means, amp_stds, lookback_fft=lookback_fft),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    model     = PaperCnnBaseline(in_channels=num_sensor_pairs * 2,
                                 num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=20)
    criterion = nn.MSELoss()

    print(f"--- 1D CNN | split={args.split} seed={args.seed} "
          f"lr={args.lr} batch={args.batch_size} epochs={args.epochs} "
          f"fft_bins={args.num_fft_bins} ---")

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
        path=checkpoint_path(args.split, "1D CNN", args.seed),
        config=vars(args),
        test_loss=test_loss,
        model=model.state_dict(),
    )
    log_result(args.split, f"1D CNN (seed={args.seed})", test_loss)


if __name__ == "__main__":
    main()