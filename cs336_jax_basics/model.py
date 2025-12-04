import torch
import os
from torch import nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
import jax
from jax import Array, Device
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx import State
import numpy.typing as npt
from collections.abc import Callable, Iterable
from typing import Optional
import math
import pickle

class Linear(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int, dtype: jnp.dtype = jnp.float32):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        # Initialize weights using the initializer
        init_fn = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std)
        weights_data = init_fn(rngs.params(), (out_features, in_features), dtype)
        self.weights = nnx.Param(weights_data)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Performs y = Wx
        """
        return jnp.einsum("...i,oi->...o", x, self.weights.get_value())


class Embedding(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, num_embeddings: int, embedding_dim: int, dtype: jnp.dtype = jnp.float32):
        super().__init__()
        std = (2 / (num_embeddings + embedding_dim)) ** 0.5
        init_fn = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std)
        weights_data = init_fn(rngs.params(), (num_embeddings, embedding_dim), dtype)
        self.weights = nnx.Param(weights_data)
    
    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        return self.weights.get_value()[token_ids]


class RMSNorm(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model: int, eps: float = 1e-5, dtype: jnp.dtype = jnp.float32):
        std = (2 / (d_model)) ** 0.5
        init_fn = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std)
        weights_data = init_fn(rngs.params(), (d_model), dtype)
        self.weights = nnx.Param(weights_data)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        d_model = x.shape[-1]
        rms_denom = (jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) ** 0.5 
        result = (x / rms_denom) * self.weights

        return result.astype(in_dtype)


class SwiGLU(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model: int, d_ff: int, dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.w1 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, dtype=dtype)
        self.w2 = Linear(rngs=rngs, in_features=d_ff, out_features=d_model, dtype=dtype)
        self.w3 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, dtype=dtype)
    
    @staticmethod
    def SiLu(x: Array) -> Array:
        return x * jax.nn.sigmoid(x)
    
    def __call__(self, x: Array) -> Array:
        product = SwiGLU.SiLu(self.w1(x)) * self.w3(x)
        return self.w2(product)


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        """Initialize the RoPE matrix.
        Args:
            theta (float): rotation constant value 
            d_k (int): embedding dimension
            max_seq_len (int): max sequence length of the model
        Returns:
            None

        Notes:
            - Creates inverse frequency tables
            - Creates position arrays 
            - Computes outer product between inverse frequency table and position
                - Rows are sequence position, column is pair position (d_k / 2)
            [
                [t(1, 1), t(1, 2), ..., t(1, d_k / 2)],
                [t(2, 1), t(2, 2), ..., t(2, d_k / 2)],
                ...
                [t(s, 1), t(s, 2), ..., t(s, d_k / 2)],
            ]
            - Block diagonal matrix of size [d_k, d_k]
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be divisible by 2"
        inv_freq = (1 / theta ** (jnp.arange(0, d_k, 2) / d_k))
        pos = jnp.arange(max_seq_len)
        angles = jnp.outer(pos, inv_freq) # [max_seq_len, d_k / 2]
        self.cos_angles = jnp.cos(angles)
        self.sin_angles = jnp.sin(angles)
    

    def __call__(self, x: Array, token_positions: Array) -> Array:
        """
        Apply rotary positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model]
            token_positions (torch.Tensor): Tensor of token positions of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Input tensor with rotary positional embeddings applied, shape [batch_size, sequence_length, d_model]

        Notes:
            - Applies RoPE by rotating pairs of dimensions using precomputed sin/cos tables
            - Even indices are rotated by cos(theta), odd indices by sin(theta)
            - Rotation is position-dependent based on token_positions
            How it works:
            - The rotation matrix is
            [
                [cos(i,k), -sin(i,k)],
                [sin(i,k), cos(i,k)]
            ]
            - When multiplying, it becomes [x1*cos(i,k) + x2*-sin(i,k), x1*sin(i,k) + x2*cos(i,k)]
            - We can group these sums by sin and cos, and move the negative to the x2 term
        """
        cos_sel = self.cos_angles[token_positions] # [B, S, d_k / 2] # (seq, pair)
        sin_sel = self.sin_angles[token_positions]
        cos_2d = jnp.repeat(cos_sel, 2, axis=-1).astype(x.dtype) # [B, S, d_k]
        sin_2d = jnp.repeat(sin_sel, 2, axis=-1).astype(x.dtype) # [B, S, d_k]
        def rotate_half(x):
            # x ~ [B, S, d_k]
            x_even = x[..., 0::2] # [B, S, d_k / 2]
            x_odd = x[..., 1::2] # [B, S, d_k / 2]
            x_rotated = jnp.stack([-x_odd, x_even], axis=-1).reshape(*x.shape[:-1], x.shape[-1])
            return x_rotated
        return x * cos_2d + rotate_half(x) * sin_2d # [B, S, d_k] * [B, S, d_k]
            

# class Softmax(torch.nn.Module):
    # def __init__(self):
        # super().__init__()


    # def forward(self, in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
        # """
        # Given a tensor of inputs, return the output of softmaxing the given `dim`
        # of the input.

        # Args:
            # in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
            # dim (int): Dimension of the `in_features` to apply softmax to.

        # Returns:
            # Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
            # softmax normalizing the specified `dim`.
        # """
        # in_features -= torch.max(in_features, dim=dim, keepdim=True)[0]
        # exp = torch.exp(in_features)
        # out = exp / torch.sum(exp, dim=dim, keepdim=True)
        # return out


def softmax(in_features: Float[Array, " ..."], dim: int) -> Float[Array, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_vals = jnp.max(in_features, axis=dim, keepdims=True)
    in_features = in_features - max_vals
    exp = jnp.exp(in_features)
    out = exp / jnp.sum(exp, axis=dim, keepdims=True)
    return out


def sdpa(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    # print(Q.shape, K.shape, V.shape)
    # [..., Q, d] x [..., K, d].T -> [..., Q, K]
    # [..., Q, K] x [V, d] -> [Q, d] since K = V

    S = jnp.einsum("... q d, ... k d -> ... q k", Q, K)
    d_k = K.shape[-1]
    S = S / (d_k) ** 0.5
    # NOTE: we use mask == 0.0 here since it takes in a float tensor, but we can also use ~mask if using booleans
    if mask is not None:
        S = jnp.where(mask == 0.0, -jnp.inf, S)
    scores = softmax(S, dim=-1)
    output = jnp.einsum("... q k, ... k d -> ... q d", scores, V)
    # output = scores @ V
    return output


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model: int, num_heads: int,
                rope_theta: float = 1e4, max_seq_len: int = 1024,
                 dtype: jnp.dtype = jnp.float32):
        """Initializes multi-head self-attention block.

        Args:
            d_model (int): input dimension 
            num_heads (int): num_heads
        """
        super().__init__()
        # Initial projection matrices
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # NOTE: num_heads * d_k == d_model, so the 'actual' dim stays constant
        self.Q_proj = Linear(rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, dtype=dtype)
        self.K_proj = Linear(rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, dtype=dtype)
        self.V_proj = Linear(rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, dtype=dtype) # d_k since d_v = d_k in this case
        self.O_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, dtype=dtype)
        # Initialize causal mask
        self.causal_mask = jnp.tril(jnp.ones((max_seq_len, max_seq_len)))
        # Intialize RoPE Module 
        self.rope = RotaryPositionalEmbedding(d_k=self.d_k, theta=rope_theta, max_seq_len=max_seq_len)


    def __call__(self, x: Array, use_rope: bool = False):
        """Forward method for Multi-Head Self Attention.

        Args:
            x (torch.Tensor[B, S, d_model]): Input tensor
            device (torch.device): Device
            dtype (torch.dtype): dtype

        Returns:
            Tensor after MHSA is applied.
        """
        B, S, d_model = x.shape
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        # NOTE: we need to first reshape, then transpose, since we want H to be the outer dim
        Q = Q.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        K = K.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        V = V.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        seq_pos = jnp.arange(0, S)
        if use_rope:
            Q = self.rope(Q, seq_pos)
            K = self.rope(K, seq_pos)
        # causal_mask = torch.tril(torch.ones(S, S, device=x.device))
        causal_mask = self.causal_mask[:S, :S] # slice existing causal mask for efficiency
        sdpa_out = sdpa(Q, K, V, causal_mask)
        # NOTE: sdpa_out ~ [B, H, S, d_k]. need out to be [B, S, d_k]
        sdpa_out = sdpa_out.swapaxes(1, 2).reshape(x.shape)
        out = self.O_proj(sdpa_out)
        return out
    

class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model: int, num_heads: int, d_ff: int, 
                 max_seq_len: int, theta: float, dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.MHA = MultiHeadSelfAttention(rngs=rngs, d_model=d_model, num_heads=num_heads, rope_theta=theta, max_seq_len=max_seq_len, dtype=dtype)
        # self.MHA2 = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.RMSNorm1 = RMSNorm(rngs=rngs, d_model=d_model, dtype=dtype)
        self.RMSNorm2 = RMSNorm(rngs=rngs, d_model=d_model, dtype=dtype)
        self.SwiGLU = SwiGLU(rngs=rngs, d_model=d_model, d_ff=d_ff, dtype=dtype)


    def __call__(self, x: Array):
        x = x + self.MHA(self.RMSNorm1(x), use_rope=True)
        x = x + self.SwiGLU(self.RMSNorm2(x))
        return x


class TransformerLM(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_model: int, num_heads: int, d_ff: int,
                theta: float, vocab_size: int, 
                context_length: int, num_layers: int, dtype: jnp.dtype = jnp.float32):
        super().__init__()
        self.layers = nnx.List([TransformerBlock(rngs, d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        self.token_embeddings = Embedding(rngs, vocab_size, d_model)
        self.ln1 = RMSNorm(rngs, d_model)
        self.linear = Linear(rngs, d_model, vocab_size)

    def __call__(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(self.ln1(x))
        # x = softmax(x, dim=-1)
        return x


def cross_entropy_loss(inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]) -> Array:
    """Computes Cross-Entropy Loss
    """
    # Stabilize logits to avoid overflow when exponentiating
    max_logits = jnp.max(inputs, axis=-1, keepdims=True)
    shifted_logits = inputs - max_logits

    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)) + max_logits
    # NOTE: this is basically log(softmax(inputs)), but reexpressed for numerical stability
    log_probs = inputs - log_sum_exp

    target_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, axis=1), axis=1).squeeze(axis=1)
    loss = -target_log_probs.mean()
    return loss


class AdamW:
    def __init__(self, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        if lr < 0:
            raise ValueError("Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.state = None
    
    def update(self, model: nnx.Module, grad_state: State):
        # Only get parameters (not all state)
        params = nnx.state(model, nnx.Param)
        if self.state is None:
            self.state = jax.tree.map(lambda p: {
                'm': jnp.zeros_like(p),
                'v': jnp.zeros_like(p),
                't': 0
            }, params)
        def update_param(param: Array, grad: Array, state: dict) -> Array:
            state['t'] += 1
            state['m'] = self.betas[0] * state['m'] + (1 - self.betas[0]) * grad
            state['v'] = self.betas[1] * state['v'] + (1 - self.betas[1]) * grad ** 2
            state['lr'] = self.lr * (1 - self.betas[1] ** state['t']) ** 0.5 / (1 - self.betas[0] ** state['t'])
            param = param - state['lr'] * state['m'] / (state['v'] ** 0.5 + self.eps)
            param = param * (1 - self.lr * self.weight_decay)
            return param

        # Use tree_map to update only leaves that exist in all three trees
        params = jax.tree.map(update_param, params, grad_state, self.state)
        nnx.update(model, params)


def get_lr_schedule(t: int, a_max: float, a_min: float, warmup_iters: int, cosine_annealing_iters: int) -> float:
    if t < warmup_iters:
        # breakpoint()
        return t / warmup_iters * a_max
    elif t <= cosine_annealing_iters:
        return a_min + (1 + math.cos((t - warmup_iters) / (cosine_annealing_iters - warmup_iters) * math.pi)) / 2 * (a_max - a_min)
    else:
        return a_min
    

def gradient_clipping(gradient_state: State, max_l2_norm: float, eps = 1e-6) -> State:
    gradients = jax.tree.leaves(gradient_state)
    total_norm = jnp.linalg.norm(jnp.concatenate([g.reshape(-1) for g in gradients]), ord=2)
    if total_norm > max_l2_norm:
        return jax.tree.map(lambda g: g * (max_l2_norm / (total_norm + eps)), gradient_state)
    return gradient_state


def get_batch(rngs: nnx.Rngs, dataset: npt.NDArray, batch_size: int, context_length: int) -> tuple[Array, Array]:
    start_indices = jax.random.randint(rngs.params(), shape=(batch_size,), minval=0, maxval=len(dataset) - context_length)
    batch = []
    for start in start_indices: 
        batch_item = jnp.array(dataset[start : start+context_length + 1])
        batch.append(batch_item)
    batch = jnp.stack(batch, axis=0)

    # B, S + 1
    train_batch = batch[:, :-1]
    target_batch = batch[:, 1:]
    assert train_batch.shape == target_batch.shape
    return (train_batch, target_batch)


def save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iteration: int,
    out,
):
    state = nnx.state(model)
    optimizer_state = optimizer.state
    obj = {
        "model": state,
        "optimizer": optimizer_state,
        "iteration": iteration,
    }

    with open(out, "wb") as f:
        pickle.dump(obj, f)

def load_checkpoint(
    src,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
):
    with open(src, "rb") as f:
        obj = pickle.load(f)
    nnx.update(model, obj["model"])
    optimizer.state = obj["optimizer"]

    return obj["iteration"]


