"""Recommender models.

The headline model is a from-scratch implementation of implicit ALS
(confidence-weighted matrix factorisation) on PyTorch: the ALS step is solved
with the conjugate gradient method, the factors stay dense and only the
interaction matrix is sparse. Baselines: Popularity, ItemKNN and PureSVD.
An alternative pairwise model, BPR-MF, is implemented as well.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["ALS", "BPR", "Popularity", "ItemKNN", "PureSVD"]


def _sparse(rows, cols, vals, shape):
    idx = torch.from_numpy(np.vstack([rows, cols]).astype(np.int64))
    v = torch.from_numpy(np.asarray(vals).astype(np.float32))
    return torch.sparse_coo_tensor(idx, v, shape).coalesce()


def _spmm(mat, dense):
    return torch.sparse.mm(mat, dense)


class ALS:
    """Implicit ALS (confidence-weighted matrix factorisation).

    Minimises the sum over observed pairs ``(u, i)`` of the confidence
    ``c_ui = 1 + alpha * r_ui`` plus the regularisation term
    ``lambda * ||u_u||^2 + ||v_i||^2`` over all factors.

    Args:
        k: Number of latent factors.
        alpha: Confidence scaling for observed pairs.
        reg: L2 regularisation.
        iters: Number of ALS iterations.
        cg_iters: Conjugate gradient iterations per ALS step.
        seed: Random seed.
        verbose: Print per-iteration progress.
    """

    def __init__(self, k=32, alpha=10.0, reg=0.05, iters=10, cg_iters=12, seed=0,
                 verbose=True):
        self.k = k
        self.alpha = alpha
        self.reg = reg
        self.iters = iters
        self.cg_iters = cg_iters
        self.seed = seed
        self.verbose = verbose
        self.U = None
        self.V = None

    @staticmethod
    def _cg(A_op, b, iters):
        """Batched conjugate gradient: solves ``A x = b`` for every row of ``b``."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = (r * r).sum(dim=1, keepdim=True)
        for _ in range(iters):
            Ap = A_op(p)
            denom = (p * Ap).sum(dim=1, keepdim=True)
            alpha = rs / (denom + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum(dim=1, keepdim=True)
            beta = rs_new / (rs + 1e-12)
            p = r + beta * p
            rs = rs_new
        return x

    def _side_step(self, rows, cols, vals, other, n_rows):
        """One ALS half-step over the rows of the matrix.

        Args:
            rows/cols/vals: Sparse interaction matrix ``Rw`` (``n_rows x n_items``).
            other: Factor matrix of the opposite side (``n_items x k``).
            n_rows: Number of rows to solve for.

        Returns:
            Factor matrix of shape ``(n_rows, k)``.
        """
        Rp = _sparse(rows, cols, np.ones_like(vals), (n_rows, other.shape[0]))
        Rw = _sparse(rows, cols, vals, (n_rows, other.shape[0]))
        # The coalesced sparse tensor stores the entries sorted by (row, col);
        # use its indices so that weights and pairs stay aligned.
        indices = Rw.indices()
        rows_t = indices[0]
        cols_t = indices[1]
        vals_t = Rw.values()
        factors = other[cols_t]                   # (nnz, k) factors of the observed items

        VtV = other.T @ other + self.reg * torch.eye(self.k)

        def A_op(x):
            # dots[u] = v_i . x_u for every observed pair (u, i)
            dots = (x[rows_t] * factors).sum(dim=1)         # (nnz,)
            g = dots * vals_t                               # r_ui * (v_i . x_u)
            Y = torch.zeros_like(x)
            Y.index_add_(0, rows_t, g.unsqueeze(1) * factors)
            # Y_u = sum_i r_ui (v_i . x_u) v_i
            return x @ VtV + self.alpha * Y

        b = _spmm(Rp, other) + self.alpha * _spmm(Rw, other)
        return self._cg(A_op, b, self.cg_iters)

    def fit(self, Rw, n_users, n_items):
        """Fit the factors.

        Args:
            Rw: Tuple ``(rows, cols, vals)`` of the positive-weight matrix.
            n_users: Number of users.
            n_items: Number of items.

        Returns:
            self
        """
        rows, cols, vals = Rw
        g = torch.Generator().manual_seed(self.seed)
        self.U = torch.randn(n_users, self.k, generator=g) * 0.01
        self.V = torch.randn(n_items, self.k, generator=g) * 0.01

        for it in range(self.iters):
            self.U = self._side_step(rows, cols, vals, self.V, n_users)
            self.V = self._side_step(cols, rows, vals, self.U, n_items)
            if self.verbose:
                print(f"    ALS iter {it + 1}/{self.iters}", flush=True)
        return self

    def score_matrix(self):
        """Return the ``(n_users, n_items)`` score matrix."""
        return self.U @ self.V.T


class BPR:
    """BPR-MF: matrix factorisation with a pairwise Bayesian Personalised Ranking loss."""

    def __init__(self, k=32, reg=1e-4, lr=0.01, epochs=8, batch=8192, n_neg=1,
                 seed=0, verbose=True):
        self.k = k
        self.reg = reg
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.n_neg = n_neg
        self.seed = seed
        self.verbose = verbose
        self.U = None
        self.V = None

    def fit(self, uidx, iidx, n_users, n_items):
        """Train the factors on positive pairs with sampled negatives.

        Args:
            uidx: User indices of the positive pairs.
            iidx: Item indices of the positive pairs.
            n_users: Number of users.
            n_items: Number of items.

        Returns:
            self
        """
        g = torch.Generator().manual_seed(self.seed)
        self.U = (torch.randn(n_users, self.k, generator=g) * 0.05).requires_grad_(True)
        self.V = (torch.randn(n_items, self.k, generator=g) * 0.05).requires_grad_(True)
        opt = torch.optim.Adam([self.U, self.V], lr=self.lr)

        u = torch.from_numpy(np.asarray(uidx).astype(np.int64))
        i = torch.from_numpy(np.asarray(iidx).astype(np.int64))
        n = u.shape[0]
        for ep in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for start in range(0, n, self.batch):
                idx = perm[start:start + self.batch]
                bu, bi = u[idx], i[idx]
                bj = torch.randint(0, n_items, (bu.shape[0] * self.n_neg,), generator=g)
                eu = self.U[bu]
                ei = self.V[bi]
                ej = self.V[bj]
                pos = (eu * ei).sum(dim=1)
                neg = (eu.repeat_interleave(self.n_neg, dim=0) * ej).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(pos - neg).mean()
                loss = loss + self.reg * (eu.pow(2).sum() + ei.pow(2).sum()
                                           + ej.pow(2).sum()) / bu.shape[0]
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * bu.shape[0]
            if self.verbose:
                print(f"    BPR epoch {ep + 1}/{self.epochs} loss={total / n:.4f}",
                      flush=True)
        return self

    def score_matrix(self):
        """Return the ``(n_users, n_items)`` score matrix."""
        return self.U @ self.V.T


class Popularity:
    """Baseline: recommend the globally most popular items."""

    def fit(self, cols, vals, n_items, n_users=1):
        """Aggregate the interaction weights per item."""
        agg = np.zeros(n_items, dtype=np.float64)
        np.add.at(agg, cols, vals.astype(np.float64))
        self.item_score = torch.from_numpy(agg.astype(np.float32))
        self.n_users = n_users
        return self

    def score_matrix(self):
        """Return the ``(n_users, n_items)`` score matrix."""
        return self.item_score.unsqueeze(0).expand(self.n_users, -1).contiguous()


class ItemKNN:
    """Item-based collaborative filtering with cosine similarity."""

    def __init__(self, topk_neighbors=200, shrink=0.0):
        self.topk_neighbors = topk_neighbors
        self.shrink = shrink

    def fit(self, rows, cols, vals, n_users, n_items):
        """Compute the item-item similarity matrix."""
        import scipy.sparse as sp

        data = np.ones_like(vals, dtype=np.float32)
        R = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        R.data[:] = 1.0
        norm = np.sqrt(np.asarray(R.multiply(R).sum(axis=0)).ravel())
        norm[norm == 0] = 1.0
        S = (R.T @ R).toarray().astype(np.float32)
        S /= (norm[:, None] * norm[None, :] + self.shrink)
        np.fill_diagonal(S, 0.0)
        if self.topk_neighbors and self.topk_neighbors < n_items:
            keep = np.argpartition(-S, self.topk_neighbors, axis=1)[:, self.topk_neighbors:]
            rows_idx = np.repeat(np.arange(n_items), keep.shape[1])
            S[rows_idx, keep.ravel()] = 0.0
        self.S = torch.from_numpy(S)
        self.Rw = _sparse(rows, cols, vals, (n_users, n_items))
        self.n_users = n_users
        return self

    def score_matrix(self):
        """Return the ``(n_users, n_items)`` score matrix."""
        return torch.sparse.mm(self.Rw, self.S)


class PureSVD:
    """PureSVD baseline: truncated SVD of the interaction matrix."""

    def __init__(self, k=32):
        self.k = k

    def fit(self, rows, cols, vals, n_users, n_items):
        """Decompose the interaction matrix with a randomised SVD."""
        from sklearn.utils.extmath import randomized_svd
        import scipy.sparse as sp

        data = np.ones_like(vals, dtype=np.float32)
        M = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.U, self.S, self.Vt = randomized_svd(M, n_components=self.k,
                                                 random_state=0)
        self.U = torch.from_numpy(self.U.astype(np.float32))
        self.V = torch.from_numpy(self.Vt.T.astype(np.float32))
        self.S = torch.from_numpy(self.S.astype(np.float32))
        return self

    def score_matrix(self):
        """Return the ``(n_users, n_items)`` score matrix."""
        return (self.U * self.S.unsqueeze(0)) @ self.V.T
