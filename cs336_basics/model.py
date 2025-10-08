import torch
from torch import nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weights = nn.Parameter(weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs y = Wx
        """
        # return x @ self.weights # this is equivalent
        return einsum(x, self.weights, "... d_in,  d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = (2 / (num_embeddings + embedding_dim)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weights = nn.Parameter(weights)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        BSZ, S -> BSZ, S, D_model 
        """
        return self.weights[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weights = torch.empty(d_model, device=device, dtype=dtype)
        std = (2 / (d_model)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        self.weights = nn.Parameter(weights)
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [4, 12, 64] w: [64]
        print(x.shape, self.weights.shape)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        d_model = x.shape[-1]
        rms_denom = (torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) ** 0.5 
        result = (x / rms_denom) * self.weights

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    @staticmethod
    def SiLu(x: torch.tensor) -> torch.tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.tensor) -> torch.Tensor:
        product = SwiGLU.SiLu(self.w1(x)) * self.w3(x)
        return self.w2(product)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device | None = None):
        """Initialize the RoPE matrix.
        Args:
            theta (float): rotation constant value 
            d_k (int): embedding dimension
            max_seq_len (int): max sequence length of the model
            device (torch.dtype): device 
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
        inv_freq = (1 / theta ** (torch.arange(0, d_k, 2) / d_k))
        pos = torch.arange(max_seq_len)
        angles = torch.outer(pos, inv_freq).to(device) # [max_seq_len, d_k / 2]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        self.register_buffer("cos_angles", cos_angles, persistent=False)
        self.register_buffer("sin_angles", sin_angles, persistent=False)
    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
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
        cos_2d = torch.repeat_interleave(cos_sel, 2, dim=-1).to(x.dtype) # [B, S, d_k]
        sin_2d = torch.repeat_interleave(sin_sel, 2, dim=-1).to(x.dtype) # [B, S, d_k]
        def rotate_half(x):
            # x ~ [B, S, d_k]
            x_even = x[..., 0::2] # [B, S, d_k / 2]
            x_odd = x[..., 1::2] # [B, S, d_k / 2]
            x_rotated = torch.stack([-x_odd, x_even], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
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


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
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
    max_vals = torch.max(in_features, dim=dim, keepdim=True)[0]
    in_features = in_features - max_vals
    exp = torch.exp(in_features)
    out = exp / torch.sum(exp, dim=dim, keepdim=True)
    return out


def sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    # print(Q.shape, K.shape, V.shape)
    # [..., Q, d] x [..., K, d].T -> [..., Q, K]
    # [..., Q, K] x [V, d] -> [Q, d] since K = V

    S = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = K.size(-1)
    S = S / (d_k) ** 0.5
    # NOTE: we use mask == 0.0 here since it takes in a float tensor, but we can also use ~mask if using booleans
    if mask is not None:
        S = S.masked_fill(mask == 0.0, float('-inf'))
    scores = softmax(S, dim=-1)
    output = einsum(scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    # output = scores @ V
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                rope_theta: float = 1e4, max_seq_len: int = 1024,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
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
        self.Q_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        self.K_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype)
        self.V_proj = Linear(d_model, self.d_k * num_heads, device=device, dtype=dtype) # d_k since d_v = d_k in this case
        self.O_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # Initialize causal mask
        self.causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device))
        # Intialize RoPE Module 
        self.rope = RotaryPositionalEmbedding(self.d_k, theta=rope_theta, max_seq_len=max_seq_len, device=device)


    def forward(self, x: torch.Tensor, use_rope: bool = False):
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
        Q = Q.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, S, self.num_heads, self.d_k).transpose(1, 2)
        seq_pos = torch.arange(0, S, device=x.device)
        if use_rope:
            Q = self.rope(Q, seq_pos)
            K = self.rope(K, seq_pos)
        # causal_mask = torch.tril(torch.ones(S, S, device=x.device))
        causal_mask = self.causal_mask[:S, :S] # slice existing causal mask for efficiency
        sdpa_out = sdpa(Q, K, V, causal_mask)
        # NOTE: sdpa_out ~ [B, H, S, d_k]. need out to be [B, S, d_k]
        sdpa_out = sdpa_out.transpose(1, 2).contiguous().reshape(x.shape)
        out = self.O_proj(sdpa_out)
        return out