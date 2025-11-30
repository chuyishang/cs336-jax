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

class Linear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        weights = jnp.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        weight_initializer = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std, dtype=dtype)
        weights = weight_initializer(rngs.params(), weights.shape, weights.dtype)
        self.weights = nnx.Param(weights)


    def __call__(self, x: Array) -> Array:
        """
        Performs y = Wx
        """
        # return x @ self.weights # this is equivalent
        return einsum(x, self.weights, "... d_in,  d_out d_in -> ... d_out")


class Embedding(nnx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        super().__init__()
        if rngs is None:
            rngs = nnx.Rngs(0)
        weights = jnp.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        std = (2 / (num_embeddings + embedding_dim)) ** 0.5
        weight_initializer = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std, dtype=dtype)
        weights = weight_initializer(rngs.params(), weights.shape, weights.dtype)
        self.weights = nnx.Param(weights)
    
    def __call__(self, token_ids: Array) -> Array:
        """
        BSZ, S -> BSZ, S, D_model 
        """
        return self.weights[token_ids]


class RMSNorm(nnx.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        weights = jnp.empty((d_model), device=device, dtype=dtype)
        std = (2 / (d_model)) ** 0.5
        weight_initializer = nnx.initializers.truncated_normal(stddev=std, lower=-3.0*std, upper=3.0*std, dtype=dtype)
        weights = weight_initializer(rngs.params(), weights.shape, weights.dtype)
        self.weights = nnx.Param(weights)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        d_model = x.shape[-1]
        rms_denom = (jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps) ** 0.5 
        result = (x / rms_denom) * self.weights

        return result.astype(in_dtype)


class SwiGLU(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        raise NotImplementedError
        # super().__init__()
        # self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    @staticmethod
    def SiLu(x: Array) -> Array:
        raise NotImplementedError
        # return x * torch.sigmoid(x)
    
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError
        # product = SwiGLU.SiLu(self.w1(x)) * self.w3(x)
        # return self.w2(product)


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: Device | None = None, rngs: nnx.Rngs | None = None):
        raise NotImplementedError
        # """Initialize the RoPE matrix.
        # Args:
        #     theta (float): rotation constant value 
        #     d_k (int): embedding dimension
        #     max_seq_len (int): max sequence length of the model
        #     device (torch.dtype): device 
        # Returns:
        #     None

        # Notes:
        #     - Creates inverse frequency tables
        #     - Creates position arrays 
        #     - Computes outer product between inverse frequency table and position
        #         - Rows are sequence position, column is pair position (d_k / 2)
        #     [
        #         [t(1, 1), t(1, 2), ..., t(1, d_k / 2)],
        #         [t(2, 1), t(2, 2), ..., t(2, d_k / 2)],
        #         ...
        #         [t(s, 1), t(s, 2), ..., t(s, d_k / 2)],
        #     ]
        #     - Block diagonal matrix of size [d_k, d_k]
        # """
        # super().__init__()
        # assert d_k % 2 == 0, "d_k must be divisible by 2"
        # inv_freq = (1 / theta ** (torch.arange(0, d_k, 2) / d_k))
        # pos = torch.arange(max_seq_len)
        # angles = torch.outer(pos, inv_freq).to(device) # [max_seq_len, d_k / 2]
        # cos_angles = torch.cos(angles)
        # sin_angles = torch.sin(angles)
        # self.register_buffer("cos_angles", cos_angles, persistent=False)
        # self.register_buffer("sin_angles", sin_angles, persistent=False)
    

    def __call__(self, x: Array, token_positions: Array) -> Array:
        raise NotImplementedError
        # """
        # Apply rotary positional embeddings to the input tensor.

        # Args:
        #     x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model]
        #     token_positions (torch.Tensor): Tensor of token positions of shape [batch_size, sequence_length]

        # Returns:
        #     torch.Tensor: Input tensor with rotary positional embeddings applied, shape [batch_size, sequence_length, d_model]

        # Notes:
        #     - Applies RoPE by rotating pairs of dimensions using precomputed sin/cos tables
        #     - Even indices are rotated by cos(theta), odd indices by sin(theta)
        #     - Rotation is position-dependent based on token_positions
        #     How it works:
        #     - The rotation matrix is
        #     [
        #         [cos(i,k), -sin(i,k)],
        #         [sin(i,k), cos(i,k)]
        #     ]
        #     - When multiplying, it becomes [x1*cos(i,k) + x2*-sin(i,k), x1*sin(i,k) + x2*cos(i,k)]
        #     - We can group these sums by sin and cos, and move the negative to the x2 term
        # """
        # cos_sel = self.cos_angles[token_positions] # [B, S, d_k / 2] # (seq, pair)
        # sin_sel = self.sin_angles[token_positions]
        # cos_2d = torch.repeat_interleave(cos_sel, 2, dim=-1).to(x.dtype) # [B, S, d_k]
        # sin_2d = torch.repeat_interleave(sin_sel, 2, dim=-1).to(x.dtype) # [B, S, d_k]
        # def rotate_half(x):
        #     # x ~ [B, S, d_k]
        #     x_even = x[..., 0::2] # [B, S, d_k / 2]
        #     x_odd = x[..., 1::2] # [B, S, d_k / 2]
        #     x_rotated = torch.stack([-x_odd, x_even], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
        #     return x_rotated
        # return x * cos_2d + rotate_half(x) * sin_2d # [B, S, d_k] * [B, S, d_k]
            

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
    raise NotImplementedError
    # """
    # Given a tensor of inputs, return the output of softmaxing the given `dim`
    # of the input.

    # Args:
    #     in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
    #     dim (int): Dimension of the `in_features` to apply softmax to.

    # Returns:
    #     Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
    #     softmax normalizing the specified `dim`.
    # """
    # max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    # in_features = in_features - max_vals
    # exp = torch.exp(in_features)
    # out = exp / torch.sum(exp, dim=dim, keepdim=True)
    # return out


def sdpa(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    raise NotImplementedError
    # # print(Q.shape, K.shape, V.shape)
    # # [..., Q, d] x [..., K, d].T -> [..., Q, K]
    # # [..., Q, K] x [V, d] -> [Q, d] since K = V

    # S = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    # d_k = K.size(-1)
    # S = S / (d_k) ** 0.5
    # # NOTE: we use mask == 0.0 here since it takes in a float tensor, but we can also use ~mask if using booleans
    # if mask is not None:
    #     S = S.masked_fill(mask == 0.0, float('-inf'))
    # scores = softmax(S, dim=-1)
    # output = einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    # # output = scores @ V
    # return output


class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int,
                rope_theta: float = 1e4, max_seq_len: int = 1024,
                 device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        raise NotImplementedError
        # """Initializes multi-head self-attention block.

        # Args:
        #     d_model (int): input dimension 
        #     num_heads (int): num_heads
        # """
        # super().__init__()
        # # Initial projection matrices
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # self.num_heads = num_heads
        # self.d_k = d_model // num_heads
        # # NOTE: num_heads * d_k == d_model, so the 'actual' dim stays constant
        # self.Q_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        # self.K_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        # self.V_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype) # d_k since d_v = d_k in this case
        # self.O_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # # Initialize causal mask
        # self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device))
        # # Intialize RoPE Module 
        # self.rope = RotaryPositionalEmbedding(self.d_k, theta=rope_theta, max_seq_len=max_seq_len, device=device)


    def __call__(self, x: Array, use_rope: bool = False):
        raise NotImplementedError
        # """Forward method for Multi-Head Self Attention.

        # Args:
        #     x (torch.Tensor[B, S, d_model]): Input tensor
        #     device (torch.device): Device
        #     dtype (torch.dtype): dtype

        # Returns:
        #     Tensor after MHSA is applied.
        # """
        # B, S, d_model = x.shape
        # Q = self.Q_proj(x)
        # K = self.K_proj(x)
        # V = self.V_proj(x)
        # # NOTE: we need to first reshape, then transpose, since we want H to be the outer dim
        # Q = Q.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # K = K.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # V = V.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        # seq_pos = torch.arange(0, S, device=x.device)
        # if use_rope:
        #     Q = self.rope(Q, seq_pos)
        #     K = self.rope(K, seq_pos)
        # # causal_mask = torch.tril(torch.ones(S, S, device=x.device))
        # causal_mask = self.causal_mask[:S, :S] # slice existing causal mask for efficiency
        # sdpa_out = sdpa(Q, K, V, causal_mask)
        # # NOTE: sdpa_out ~ [B, H, S, d_k]. need out to be [B, S, d_k]
        # sdpa_out = sdpa_out.transpose(1, 2).contiguous().reshape(x.shape)
        # out = self.O_proj(sdpa_out)
        # return out
    

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 max_seq_len: int, theta: float, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        raise NotImplementedError
        # super().__init__()
        # self.MHA = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        # # self.MHA2 = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        # self.RMSNorm1 = RMSNorm(d_model)
        # self.RMSNorm2 = RMSNorm(d_model)
        # self.SwiGLU = SwiGLU(d_model, d_ff)


    def __call__(self, x: Array):
        raise NotImplementedError
        # x = x + self.MHA(self.RMSNorm1(x), use_rope=True)
        # x = x + self.SwiGLU(self.RMSNorm2(x))
        # return x


class TransformerLM(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                theta: float, vocab_size: int, 
                context_length: int, num_layers: int, device: Device | None = None, dtype: jnp.dtype | None = None, rngs: nnx.Rngs | None = None):
        raise NotImplementedError
        # super().__init__()
        # self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        # self.token_embeddings = Embedding(vocab_size, d_model)
        # self.ln1 = RMSNorm(d_model)
        # self.linear = Linear(d_model, vocab_size)

    def __call__(self, x):
        raise NotImplementedError
        # x = self.token_embeddings(x)
        # for layer in self.layers:
        #     x = layer(x)
        # x = self.linear(self.ln1(x))
        # # x = softmax(x, dim=-1)
        # return x


def cross_entropy_loss(inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]) -> Array:
    raise NotImplementedError
    # """Computes Cross-Entropy Loss
    # """
    # # Stabilize logits to avoid overflow when exponentiating
    # max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    # shifted_logits = inputs - max_logits

    # log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)) + max_logits
    # # NOTE: this is basically log(softmax(inputs)), but reexpressed for numerical stability
    # log_probs = inputs - log_sum_exp

    # target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # loss = -target_log_probs.mean()
    # return loss


# SGD Example
from collections.abc import Callable, Iterable
from typing import Optional
import math
class SGD:
    def __init__(self, params, lr=1e-3):
        raise NotImplementedError
        # if lr < 0: 
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # defaults = {"lr": lr}
        # super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        raise NotImplementedError
        # loss = None if closure is None else closure()
        # for group in self.param_groups:
        #     lr = group["lr"]
        #     for p in group["params"]:
        #         if p.grad is None:
        #             continue
        #             
        #         state = self.state[p]
        #     t = state.get("t", 0)
        #     grad = p.grad.data
        #     p.data -= lr / math.sqrt(t + 1) * grad
        #     state["t"] = t + 1
        # return loss


class AdamW:
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps = 1e-8):
        raise NotImplementedError
        # if lr < 0:
        #     raise ValueError("Invalid learning rate: {lr}")
        # if not 0.0 <= betas[0] < 1.0:
        #     raise ValueError(f"Invalid beta1 value: {betas[0]}")
        # if not 0.0 <= betas[1] < 1.0:
        #     raise ValueError(f"Invalid beta2 value: {betas[1]}")
        # defaults = {"t": 0, "lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}


        # super().__init__(params, defaults)
    
    def step(self, closure=None):
        raise NotImplementedError
        # loss = None if closure is None else closure()
        # for group in self.param_groups:
        #     lr = group["lr"]
        #     beta1 = group["beta1"]
        #     beta2 = group["beta2"]
        #     weight_decay = group["weight_decay"]
        #     eps = group["eps"]
        #     for p in group["params"]:
        #         if p.grad is None:
        #             continue
        #         state = self.state[p]
        #         if len(state) == 0:
        #             state['m'] = torch.zeros_like(p.data)
        #             state['v'] = torch.zeros_like(p.data)
        #             state['t'] = 0
        #         state['t'] += 1
        #         grad = p.grad.data
        #         state['m'] = beta1 * state['m'] + (1 - beta1) * grad
        #         state['v'] = beta2 * state['v'] + (1 - beta2) * grad ** 2
        #         state['lr'] = lr * (1 - beta2 ** state['t']) ** 0.5 / (1 - beta1 ** state['t'])
        #         p.data -= state['lr'] * state['m'] / (state['v'] ** 0.5 + eps)
        #         p.data = p.data * (1 -  lr * weight_decay)
        # return loss


def get_lr_schedule(t, a_max, a_min, warmup_iters, cosine_annealing_iters):
    raise NotImplementedError
    # if t < warmup_iters:
    #     return t / warmup_iters * a_max
    # elif t <= cosine_annealing_iters:
    #     return a_min + (1 + math.cos((t - warmup_iters) / (cosine_annealing_iters - warmup_iters) * math.pi)) / 2 * (a_max - a_min)
    # else:
    #     return a_min
    

def gradient_clipping(parameters: Iterable, max_l2_norm: float, eps = 1e-6) -> None:
    raise NotImplementedError
    # total_norm = torch.norm(torch.stack([p.grad.data for p in parameters if p.grad is not None]), p=2)
    # if total_norm > max_l2_norm:
    #     for p in parameters:
    #         if p.grad is None:
    #             continue
    #         p.grad.data = p.grad.data * (max_l2_norm / (total_norm + eps))


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Array, Array]:
    raise NotImplementedError
    # start_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    # batch = []
    # for start in start_indices: 
    #     batch_item = torch.tensor(dataset[start : start+context_length + 1])
    #     batch.append(batch_item)
    # batch = torch.stack(batch, dim=0).to(device)
    # # B, S + 1
    # train_batch = batch[:, :-1]
    # target_batch = batch[:, 1:]
    # assert train_batch.shape == target_batch.shape
    # return (train_batch, target_batch)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
):
    raise NotImplementedError
    # state = model.state_dict()
    # optimizer_state = optimizer.state_dict()
    # obj = {
    #     "model": state,
    #     "optimizer": optimizer_state,
    #     "iteration": iteration,
    # }

    # torch.save(obj, out)

def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    raise NotImplementedError
    # obj = torch.load(src)
    # model.load_state_dict(obj["model"])
    # optimizer.load_state_dict(obj["optimizer"])

    # return obj["iteration"]


