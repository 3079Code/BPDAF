"""Run a small, dataset-free optimizer and checkpoint smoke test.

The MSE objective below is a synthetic debugging objective, NOT a replacement
for the detector losses in the paper. No detection accuracy is measured here.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from bpdaf import BPDAFFuser, ConvFuser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1788189656)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional checkpoint file, e.g. runs/smoke.pt")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    torch.manual_seed(args.seed)
    config = dict(in_channels=[4, 8], out_channels=8, bev_range=[-6.0, -4.0, 6.0, 4.0])
    reference = ConvFuser(config["in_channels"], config["out_channels"]).to(args.device).eval()
    model = BPDAFFuser.from_reference(reference, config["bev_range"])
    features = [
        torch.randn(2, 4, 8, 12, device=args.device),
        torch.randn(2, 8, 8, 12, device=args.device),
    ]
    with torch.no_grad():
        output, auxiliary = model.forward_with_aux(features)
        initial_equal = torch.equal(output, reference(features))
        if not initial_equal or torch.count_nonzero(auxiliary["residual"]).item() != 0:
            raise RuntimeError("initial reference equivalence failed")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    target = torch.rand_like(output)
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(features), target)
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic training produced a nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        final_output, auxiliary = model.forward_with_aux(features)
        residual_norm = auxiliary["residual"].norm().item()
        if residual_norm == 0 or not torch.isfinite(final_output).all():
            raise RuntimeError("residual did not learn a finite correction")
    report = {
        "purpose": "synthetic smoke test; not a detection benchmark",
        "torch": torch.__version__,
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "initial_reference_equal": initial_equal,
        "synthetic_losses": losses,
        "final_residual_norm": residual_norm,
        "output_shape": list(final_output.shape),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format_version": 1, "config": config, "state_dict": model.state_dict()},
            args.output,
        )
        restored = torch.load(args.output, map_location=args.device, weights_only=True)
        reloaded = BPDAFFuser(**restored["config"]).to(args.device).eval()
        reloaded.load_state_dict(restored["state_dict"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(reloaded(features), final_output, rtol=0, atol=0)
        report["checkpoint_roundtrip_equal"] = True
        report["checkpoint"] = str(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
