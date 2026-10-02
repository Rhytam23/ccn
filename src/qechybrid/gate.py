"""Learned fast-path gate (the AI pre-decoder stage of the hybrid pipeline).

A small MLP reads the raw detector syndrome and predicts the logical flip. A
calibrated confidence threshold decides which shots the network may answer on its
own; everything else is offloaded to the exact global decoder (MWPM). The threshold
is calibrated on held-out data to limit the extra logical errors it adds; this is an empirical check on the tested cases, not a guarantee.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GateMLP(nn.Module):
    def __init__(self, n_det: int, n_obs: int = 1, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_det, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_obs),
        )

    def forward(self, x):
        return self.net(x)


class Gate:
    def __init__(self, n_det: int, n_obs: int = 1, hidden: int = 256, device: str = "cpu"):
        self.device = device
        self.model = GateMLP(n_det, n_obs, hidden).to(device)
        self.threshold = float("inf")  # |logit| confidence; accept nothing until calibrated

    def fit(self, dets: np.ndarray, obs: np.ndarray, epochs: int = 6, batch: int = 1024, lr: float = 2e-3, seed: int = 0):
        torch.manual_seed(seed)
        X = torch.as_tensor(dets, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            for i in range(0, len(X), batch):
                idx = perm[i : i + batch]
                opt.zero_grad()
                lossf(self.model(X[idx]), Y[idx]).backward()
                opt.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def logits(self, dets) -> torch.Tensor:
        x = torch.as_tensor(dets, dtype=torch.float32, device=self.device)
        return self.model(x)

    def proba(self, dets) -> torch.Tensor:
        return torch.sigmoid(self.logits(dets))

    def calibrate(self, dets, obs, matching_pred, max_extra_error_frac: float = 0.05) -> float:
        """Lowest |logit| confidence threshold whose accepted set costs at most
        `max_extra_error_frac` x (matching's total errors) extra logical errors."""
        z = self.logits(dets).cpu().numpy()[:, 0]
        y = obs[:, 0].astype(bool)
        nn_pred = z > 0
        mm_pred = matching_pred[:, 0].astype(bool)
        conf = np.abs(z)  # logit magnitude: unlike probabilities it does not saturate at 1.0
        order = np.argsort(-conf, kind="stable")
        extra = np.cumsum((nn_pred[order] != y[order]).astype(int) - (mm_pred[order] != y[order]).astype(int))
        budget = max_extra_error_frac * max(1, int((mm_pred != y).sum()))
        ok = np.nonzero(extra <= budget)[0]
        self.threshold = float("inf") if len(ok) == 0 else float(conf[order[ok[-1]]])
        return self.threshold
