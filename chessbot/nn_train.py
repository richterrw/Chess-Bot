"""Train the neural move predictor on data/nn_data.npz.

Phase 1 learns from every move in the games (Robbie's and his opponents'),
so the network picks up how people at this level play. Phase 2 fine-tunes
on Robbie's own moves only. Accuracy is measured on Robbie's moves in games
held out from training.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .nn_model import MoveNet, norm_rating


def batches(idx, X, Y, M, bs, shuffle, rng):
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        planes = np.unpackbits(X[j], axis=1)[:, :21 * 64].reshape(-1, 21, 8, 8)
        yield (torch.from_numpy(planes.astype(np.float32)),
               torch.from_numpy(M[j, 0].astype(np.int64)),
               torch.from_numpy(norm_rating(M[j, 1].astype(np.float32))),
               torch.from_numpy(M[j, 2].astype(np.float32)),
               torch.from_numpy(Y[j].astype(np.int64)))


def evaluate(net, idx, X, Y, M):
    net.eval()
    correct = top3 = n = 0
    loss = 0.0
    with torch.no_grad():
        for p, tc, r, rb, y in batches(idx, X, Y, M, 2048, False, None):
            out = net(p, tc, r, rb)
            loss += F.cross_entropy(out, y, reduction="sum").item()
            t3 = out.topk(3, dim=1).indices
            correct += (t3[:, 0] == y).sum().item()
            top3 += (t3 == y[:, None]).any(1).sum().item()
            n += len(y)
    net.train()
    return correct / n, top3 / n, loss / n


def run_phase(net, opt, train_idx, test_idx, data, epochs, lr, bs, rng, label, out):
    X, Y, M = data
    steps = epochs * math.ceil(len(train_idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for p, tc, r, rb, y in batches(train_idx, X, Y, M, bs, True, rng):
            loss = F.cross_entropy(net(p, tc, r, rb), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        acc, acc3, tl = evaluate(net, test_idx, X, Y, M)
        print(f"[{label}] epoch {ep + 1}/{epochs}: train loss {tot / n:.3f}, held-out Robbie moves: "
              f"top-1 {acc:.1%}, top-3 {acc3:.1%}, loss {tl:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        torch.save(net.state_dict(), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/nn_data.npz")
    ap.add_argument("--out", default="data/movenet.pt")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--finetune-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    d = np.load(args.data)
    X, Y, M, T = d["X"], d["Y"], d["M"], d["T"]
    robbie = M[:, 2] == 1
    train_all = np.where(~T)[0]
    train_robbie = np.where(~T & robbie)[0]
    test_robbie = np.where(T & robbie)[0]
    print(f"train {len(train_all)} positions ({len(train_robbie)} Robbie), test {len(test_robbie)} Robbie moves", flush=True)
    rng = np.random.default_rng(0)
    net = MoveNet(args.channels, args.blocks)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    run_phase(net, opt, train_all, test_robbie, (X, Y, M), args.epochs, 2e-3, args.batch, rng, "all moves", args.out)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    run_phase(net, opt, train_robbie, test_robbie, (X, Y, M), args.finetune_epochs, 3e-4, args.batch, rng, "Robbie only", args.out)


if __name__ == "__main__":
    main()
