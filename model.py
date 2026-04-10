"""
NumPy MLP for MNIST / CIFAR-10 (2 hidden layers).
Clean forward + backprop, no external ML libraries needed.
All federated/compression/latency logic unchanged.
"""

import numpy as np


def _relu(x):    return np.maximum(0.0, x)
def _relu_d(x):  return (x > 0).astype(np.float32)
def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x); return e / (e.sum(axis=1, keepdims=True) + 1e-12)
def _xe(p, y):
    n = y.shape[0]
    return float(-np.log(p[np.arange(n), y] + 1e-12).mean())
def _he(fan_in, fan_out):
    return (np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)).astype(np.float32)


class MLP:
    """
    MNIST:   784 → 256 → 128 → 10
    CIFAR10: 3072 → 512 → 256 → 10
    """
    def __init__(self, dataset="MNIST"):
        self.dataset = dataset
        if dataset == "MNIST":
            d0, d1, d2, d3 = 784, 256, 128, 10
        else:
            d0, d1, d2, d3 = 3072, 512, 256, 10
        self.W1 = _he(d0, d1); self.b1 = np.zeros(d1, np.float32)
        self.W2 = _he(d1, d2); self.b2 = np.zeros(d2, np.float32)
        self.W3 = _he(d2, d3); self.b3 = np.zeros(d3, np.float32)
        self._c = {}

    # ── Params ───────────────────────────────────────────────────────────────
    def get_params(self):
        return np.concatenate([v.ravel() for v in
            [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]])

    def set_params(self, flat):
        offs, attrs = 0, [("W1",self.W1),("b1",self.b1),("W2",self.W2),
                           ("b2",self.b2),("W3",self.W3),("b3",self.b3)]
        for name, arr in attrs:
            n = arr.size
            setattr(self, name, flat[offs:offs+n].reshape(arr.shape).astype(np.float32))
            offs += n

    def num_params(self): return self.get_params().size

    def clone(self):
        m = MLP(self.dataset); m.set_params(self.get_params().copy()); return m

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        # x: (N, C, H, W) or (N, D) — flatten
        x = x.reshape(x.shape[0], -1).astype(np.float32)
        z1 = x  @ self.W1 + self.b1; a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        self._c = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2)
        return z3.astype(np.float32)

    # ── Backward ─────────────────────────────────────────────────────────────
    def backward(self, logits, labels):
        c = self._c; N = labels.shape[0]
        p = _softmax(logits); loss = _xe(p, labels)
        dz3 = p.copy(); dz3[np.arange(N), labels] -= 1; dz3 /= N
        dW3 = c["a2"].T @ dz3; db3 = dz3.sum(0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * _relu_d(c["z2"])
        dW2 = c["a1"].T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_d(c["z1"])
        dW1 = c["x"].T @ dz1;  db1 = dz1.sum(0)
        return dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3), loss

    def apply_grads(self, grads, lr, momentum=0.9, wd=1e-4, vel=None):
        if vel is None: vel = {k: np.zeros_like(v) for k,v in grads.items()}
        for name in ["W1","b1","W2","b2","W3","b3"]:
            g = grads[name] + wd * getattr(self, name)
            vel[name] = momentum * vel[name] + g
            setattr(self, name, getattr(self, name) - lr * vel[name])
        return vel


# ── Public helpers (same interface as before) ─────────────────────────────────
def get_model(dataset, device="cpu"): return MLP(dataset)
def count_parameters(model):         return model.num_params()
def model_size_mb(model, bits=32):   return (model.num_params() * bits) / 8e6
def get_flat_params(model):          return model.get_params()
def set_flat_params(model, flat):    model.set_params(flat)
def clone_model(model):              return model.clone()
