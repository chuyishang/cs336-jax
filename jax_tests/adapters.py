from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from jax import Array
import jax.numpy as jnp
from flax import nnx
from flax.nnx import State

import numpy.typing as npt

from cs336_jax_basics import train_bpe
from cs336_jax_basics import tokenizer
from cs336_jax_basics import model

from .common import create_linear_layer_state, create_embedding_state, create_rmsnorm_state, create_swiglu_state, create_mha_state
from .conftest import tensor_to_array

def perpetual_rngs(seed: int):
    rngs = nnx.Rngs(seed)
    def create_wrapper(function: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs, rngs=rngs)
        return wrapper
    return create_wrapper

@perpetual_rngs(0)
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Array, " d_out d_in"],
    in_features: Float[Array, " ... d_in"],
    rngs: nnx.Rngs
) -> Float[Array, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Array, "... d_out"]: The transformed output of your linear module.
    """
    linear_layer = model.Linear(rngs=rngs, in_features=d_in, out_features=d_out)
    linear_layer_state = create_linear_layer_state(weights)
    nnx.update(linear_layer, linear_layer_state)
    return linear_layer(in_features)

@perpetual_rngs(1)
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Array, " vocab_size d_model"],
    token_ids: Int[Array, " ..."],
    rngs: nnx.Rngs
) -> Float[Array, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Array, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Array, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Array, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = model.Embedding(rngs=rngs, num_embeddings=vocab_size, embedding_dim=d_model)
    embedding_state = create_embedding_state(weights)
    nnx.update(embedding, embedding_state)
    return embedding(token_ids)



@perpetual_rngs(2)
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Array, " d_ff d_model"],
    w2_weight: Float[Array, " d_model d_ff"],
    w3_weight: Float[Array, " d_ff d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs
) -> Float[Array, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Array, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Array, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Array, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Array, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Array, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = model.SwiGLU(rngs=rngs, d_model=d_model, d_ff=d_ff)
    swiglu_state = create_swiglu_state(w1_weight, w2_weight, w3_weight)
    nnx.update(swiglu, swiglu_state)
    # swiglu.w1.weights.data = w1_weight
    # swiglu.w2.weights.data = w2_weight
    # swiglu.w3.weights.data = w3_weight
    return swiglu(in_features)



def run_scaled_dot_product_attention(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Array, " ... queries d_k"]): Query tensor
        K (Float[Array, " ... keys d_k"]): Key tensor
        V (Float[Array, " ... values d_v"]): Values tensor
        mask (Float[Array, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Array, " ... queries d_v"]: Output of SDPA
    """
    return model.sdpa(Q, K, V, mask)


@perpetual_rngs(3)
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs
) -> Float[Array, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Array, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Array, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Array, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Array, "d_model d_v"]): Weights for the output projection
        in_features (Float[Array, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Array, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA = model.MultiHeadSelfAttention(rngs=rngs, d_model=d_model, num_heads=num_heads)
    MHA_state = create_mha_state(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    nnx.update(MHA, MHA_state)
    out = MHA(in_features, use_rope=False)
    return out
    

@perpetual_rngs(4)
def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs,
    token_positions: Int[Array, " ... sequence_length"] | None = None,
) -> Float[Array, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Array, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Array, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Array, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Array, "d_model d_v"]): Weights for the output projection
        in_features (Float[Array, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Array, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Array, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    MHA = model.MultiHeadSelfAttention(rngs=rngs, d_model=d_model, num_heads=num_heads, rope_theta=theta, max_seq_len=max_seq_len)
    MHA_state = create_mha_state(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    nnx.update(MHA, MHA_state)
    out = MHA(in_features, use_rope=True)
    return out


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Array, " ... sequence_length d_k"],
    token_positions: Int[Array, " ... sequence_length"],
) -> Float[Array, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Array, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Array, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Array, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = model.RotaryPositionalEmbedding(d_k, theta, max_seq_len)
    result = rope(in_query_or_key, token_positions)
    return result


@perpetual_rngs(5)
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Array],
    in_features: Float[Array, " batch sequence_length d_model"],
    rngs: nnx.Rngs
) -> Float[Array, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Array]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Array, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Array, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    TransformerBlock = model.TransformerBlock(rngs=rngs, d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta)
    TransformerBlock_state = State(
        {
            'MHA': create_mha_state(weights['attn.q_proj.weight'], weights['attn.k_proj.weight'], weights['attn.v_proj.weight'], weights['attn.output_proj.weight']),
            'SwiGLU': create_swiglu_state(weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight']),
            'RMSNorm1': create_rmsnorm_state(weights['ln1.weight']),
            'RMSNorm2': create_rmsnorm_state(weights['ln2.weight']),
        })
    nnx.update(TransformerBlock, TransformerBlock_state)

    out = TransformerBlock(in_features)
    return out


@perpetual_rngs(6)
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Array],
    in_indices: Int[Array, " batch_size sequence_length"],
    rngs: nnx.Rngs
) -> Float[Array, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Array]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Array, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Array, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    # Create the TransformerLM model
    transformer_lm = model.TransformerLM(
        rngs=rngs,
        d_model=d_model,
        num_heads=num_heads, 
        d_ff=d_ff,
        theta=rope_theta,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers
    )
    
    # Create state dict mapping from reference weights to our model structure
    state_dict = {}
    
    # Token embeddings
    state_dict['token_embeddings'] = create_embedding_state(weights['token_embeddings.weight'])
    
    # Transformer layers
    for layer_idx in range(num_layers):
        layer_prefix = f'layers.{layer_idx}'
        if "layers" not in state_dict:
            state_dict['layers'] = {}
        if layer_idx not in state_dict['layers']:
            state_dict['layers'][layer_idx] = {}
        
        layer_state_dict = state_dict['layers'][layer_idx]

        # Attention weights
        layer_state_dict['MHA'] = (
            create_mha_state(weights[f'{layer_prefix}.attn.q_proj.weight'], weights[f'{layer_prefix}.attn.k_proj.weight'], weights[f'{layer_prefix}.attn.v_proj.weight'], weights[f'{layer_prefix}.attn.output_proj.weight'])
        )

        # FFN weights
        layer_state_dict['SwiGLU'] = (
            create_swiglu_state(weights[f'{layer_prefix}.ffn.w1.weight'], weights[f'{layer_prefix}.ffn.w2.weight'], weights[f'{layer_prefix}.ffn.w3.weight'])
        )

        # Norm weights
        layer_state_dict['RMSNorm1'] = (
            create_rmsnorm_state(weights[f'{layer_prefix}.ln1.weight'])
        )
        layer_state_dict['RMSNorm2'] = (
            create_rmsnorm_state(weights[f'{layer_prefix}.ln2.weight'])
        )
    
    # Final layer norm and output projection
    state_dict['ln1'] = create_rmsnorm_state(weights['ln_final.weight'])
    state_dict['linear'] = create_linear_layer_state(weights['lm_head.weight'])
    
    # Load the state dict
    nnx.update(transformer_lm, state_dict)
    
    # Run forward pass
    return transformer_lm(in_indices)


@perpetual_rngs(7)
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Array, " d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs
) -> Float[Array, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Array,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = model.RMSNorm(rngs=rngs, d_model=d_model, eps=eps)
    rms_norm_state = create_rmsnorm_state(weights)
    nnx.update(rms_norm, rms_norm_state)
    return rms_norm(in_features)  


def run_silu(in_features: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Array,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return model.SwiGLU.SiLu(in_features)


@perpetual_rngs(8)
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, rngs: nnx.Rngs
) -> tuple[Array, Array]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.

    Returns:
        Tuple of jax.Arrays of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return model.get_batch(rngs, dataset, batch_size, context_length)


def run_softmax(in_features: Float[Array, " ..."], dim: int) -> Float[Array, " ..."]:
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
    # softmax = model.Softmax()
    return model.softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]
) -> Float[Array, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Array, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Array, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Array, ""]: The average cross-entropy loss across examples.
    """
    return model.cross_entropy_loss(inputs, targets)


def run_gradient_clipping(gradient_state: State, max_l2_norm: float) -> State:
    """Given a gradient state, clip the gradients to have l2 norm at most max_l2_norm.

    Args:
        gradient_state (State): collection of gradients.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    Returns:
        State: the clipped gradient state.
    """
    return model.gradient_clipping(gradient_state, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return model.AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return model.get_lr_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    from cs336_jax_basics import model as model_module
    return model_module.save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    from cs336_jax_basics import model as model_module
    return model_module.load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return tokenizer.Tokenizer(vocab, merges, special_tokens)
    # return ref.Tokenizer(vocab, merges, special_tokens)



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe.train_bpe(input_path, vocab_size, special_tokens)
