"""Export a trained MoveNet for the browser (web/movenet.js reads it).

Batch norm is folded into the convolutions. The output is one JSON file with
the shape and the weights as base64 little-endian float32, in this order:
stem, each block's two convs, head. Each conv is weights [out][in][k][k]
followed by bias [out].
"""
import base64
import argparse
import json

import numpy as np
import torch

from .nn_model import MoveNet


def fold(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    if bn is None:
        return w, conv.bias.detach().numpy().astype(np.float64)
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    return w * scale[:, None, None, None], b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="data/movenet.pt")
    ap.add_argument("--out", default="web/data/movenet.json")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    args = ap.parse_args()
    net = MoveNet(args.channels, args.blocks)
    net.load_state_dict(torch.load(args.model))
    net.eval()
    layers = [fold(net.stem, net.stem_bn)]
    for blk in net.blocks:
        layers += [fold(blk.c1, blk.b1), fold(blk.c2, blk.b2)]
    layers.append(fold(net.head, None))
    parts = []
    for w, b in layers:
        parts += [w.astype("<f4").ravel(), b.astype("<f4").ravel()]
    blob = np.concatenate(parts)
    meta = {"channels": args.channels, "blocks": args.blocks, "inputs": int(net.stem.in_channels),
            "floats": int(blob.size), "weights": base64.b64encode(blob.tobytes()).decode("ascii")}
    with open(args.out, "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {args.out}: {blob.size} floats ({blob.size * 4 / 1e6:.1f} MB), {meta}")


if __name__ == "__main__":
    main()
