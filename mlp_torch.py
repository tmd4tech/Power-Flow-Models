"""PyTorch MLP regressor, replacing sklearn's MLPRegressor.

Why: sklearn's early stopping halts when its validation R^2 fails to improve by
tol=1e-4 for n_iter_no_change epochs. With 236 outputs (case118) that triggered
after a few epochs: MLPs trained on the intact grid had 6 degree angle errors on
the INTACT grid, 30x worse than a linear model.

What stays the same as before: hidden layer sizes and learning rates (the
tuning grid), ReLU, Adam, L2 penalty alpha=1e-4, and a random 10% of the
training samples held out to decide when to stop.

What changes: training continues while validation MSE improves AT ALL, stops
after PATIENCE epochs without improvement (max MAX_EPOCHS), and restores the
best weights seen. Runs on the GPU when one is available.

LINEAR PATH: output = linear(x) + mlp(x). The linear part is a FIXED ridge
solution passed in by the caller (the regime's own ridge model, penalty chosen
by outage-grouped CV); the MLP learns only the nonlinear remainder. The MLP's
last layer starts at zero, so training starts AT the linear model, and the
start is evaluated as a candidate for the best weights: the result cannot be
worse than the linear model on the validation split. The linear part is not
trained, because a weakly regularized linear map extrapolates badly on unseen
outages (0.5 p.u. errors in Phase 3).

Device: CUDA if available, else CPU. Override with the environment variable
GRID_DEVICE=cpu (or cuda). GPU results can differ from CPU results in the last
decimal places; use one device type for all runs of a comparison.
"""

import os

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device(os.environ.get("GRID_DEVICE")
                      or ("cuda" if torch.cuda.is_available() else "cpu"))

MAX_EPOCHS = 1000
PATIENCE = 30
BATCH = 256
VAL_FRACTION = 0.1
PREDICT_CHUNK = 8192


class TorchMLP:
    def __init__(self, hidden_layer_sizes=(128, 128), learning_rate_init=1e-3,
                 alpha=1e-4, random_state=0, linear=None, **_ignored):
        """linear: (coef (n_out, n_in), intercept (n_out,)) of the fixed linear path,
        or None for a plain MLP."""
        self.linear = linear
        self.hidden = tuple(hidden_layer_sizes)
        self.lr = learning_rate_init
        self.alpha = alpha
        self.seed = random_state

    def _build(self, n_in, n_out):
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        last = nn.Linear(d, n_out)
        if self.linear is not None:
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # start at the linear model
        layers.append(last)
        return nn.Sequential(*layers).to(DEVICE)

    def _forward(self, x):
        out = self.net(x)
        if self.linear is not None:
            out = out + x @ self.W.T + self.b
        return out

    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * VAL_FRACTION)))
        va, tr = perm[:n_val], perm[n_val:]
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Yt = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
        self.net = self._build(X.shape[1], Y.shape[1])
        if self.linear is not None:
            self.W = torch.tensor(np.asarray(self.linear[0]), dtype=torch.float32, device=DEVICE)
            self.b = torch.tensor(np.asarray(self.linear[1]), dtype=torch.float32, device=DEVICE)
        # Adam + L2 penalty. sklearn's alpha penalizes alpha*||W||^2/(2n);
        # weight_decay = alpha/n_train is the closest Adam equivalent. Not exact:
        # PyTorch also decays biases and the loss scale differs by 2x. With
        # alpha = 1e-4 the penalty is negligible either way.
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.alpha / max(1, len(tr)))
        tr_t = torch.tensor(tr, device=DEVICE)
        self.net.eval()
        with torch.no_grad():                        # the starting point is a candidate
            best = float(((self._forward(Xt[va]) - Yt[va]) ** 2).mean())
        best_state = {k: p.detach().clone() for k, p in self.net.state_dict().items()}
        bad, epoch = 0, -1
        for epoch in range(MAX_EPOCHS):
            self.net.train()
            order = tr_t[torch.randperm(len(tr), device=DEVICE)]
            for k in range(0, len(order), BATCH):
                b = order[k:k + BATCH]
                opt.zero_grad()
                loss = ((self._forward(Xt[b]) - Yt[b]) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(((self._forward(Xt[va]) - Yt[va]) ** 2).mean())
            if v < best:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.n_iter_ = epoch + 1
        self.best_val_mse_ = best
        return self

    def predict(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(X), PREDICT_CHUNK):
                xb = torch.tensor(X[k:k + PREDICT_CHUNK], dtype=torch.float32, device=DEVICE)
                out.append(self._forward(xb).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.net[-1].out_features))
