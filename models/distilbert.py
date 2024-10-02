import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #
DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'distilbert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-pytorch_model.bin",
    'distilbert-base-uncased-distilled-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-pytorch_model.bin"
}


def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    if is_deepspeed_zero3_enabled():
        import deepspeed

        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


class Embeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.is_causal = False

        # Have an even number of multi heads that divide the dimensions
        if self.dim % self.n_heads != 0:
            # Raise value errors for even multi-head attention nodes
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads: Set[int] = set()
        self.attention_head_size = self.dim // self.n_heads

    def prune_heads(self, heads: List[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class DistilBertFlashAttention2(MultiHeadSelfAttention):
    """
    DistilBert flash attention module. This module inherits from `MultiHeadSelfAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        batch_size, q_length, dim = query.size()

        dim_per_head = self.dim // self.n_heads

        def reshape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(batch_size, -1, self.n_heads, dim_per_head)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = reshape(self.q_lin(query))
        key_states = reshape(self.k_lin(key))
        value_states = reshape(self.v_lin(value))

        attn_dropout = self.config.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_lin.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_weights = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            mask,
            q_length,
            dropout=attn_dropout,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_weights_reshaped = attn_weights.reshape(batch_size, q_length, self.n_heads * dim_per_head)
        attn_output = self.out_lin(attn_weights_reshaped)

        if output_attentions:
            return (attn_output, attn_weights)
        else:
            return (attn_output,)


class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


DISTILBERT_ATTENTION_CLASSES = {
    "eager": MultiHeadSelfAttention,
    "flash_attention_2": DistilBertFlashAttention2,
}


class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        # Have an even number of Configure multi-heads
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        self.attention = DISTILBERT_ATTENTION_CLASSES[config._attn_implementation](config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            if type(sa_output) is not tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")

            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output



class MaskLayer(nn.Module):
    def __init__(self, lower_bound, upper_bound, replacement_values):
        super(MaskLayer, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.replacement_values = replacement_values

 

    def forward(self, x):
        lower_bound = self.lower_bound.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        upper_bound = self.upper_bound.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        replacement_values = self.replacement_values.to(dtype=x.dtype, device=x.device).view(1, 1, -1)

 

        mask = (x >= lower_bound) & (x <= upper_bound)
        x = torch.where(mask, replacement_values, x)
        return x
    
    def set_perms(self,lower_bound, upper_bound, replacement_values):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.replacement_values = replacement_values


class Transformer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.gradient_checkpointing = False
        self.masking_layer = torch.ones(config.hidden_size).to("cuda")
        self.m_layer = config.m_layer
        self.mask_layer = MaskLayer(torch.tensor(float('inf')), torch.tensor(float('-inf')), torch.tensor(0.0))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )
            if(i == self.m_layer):
                hidden_state = self.mask_layer(layer_outputs[-1])
                # hidden_state = layer_outputs[-1]*self.masking_layer
            else:
                hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )
        
        
### INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL ###
class DistilBertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def __init__(self, *inputs, **kwargs):
        super(DistilBertPreTrainedModel, self).__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, nn.Embedding):
            if module.weight.requires_grad:
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class DistilBertModel(DistilBertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(DistilBertModel, self).__init__(config)

        self.embeddings = Embeddings(config)   # Embeddings
        self.transformer = Transformer(config) # Encoder

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids)   # (bs, seq_length, dim)
        tfmr_output = self.transformer(x=embedding_output,
                                       attn_mask=attention_mask,
                                       head_mask=head_mask)
        hidden_state = tfmr_output[0]
        output = (hidden_state, ) + tfmr_output[1:]

        return output # last-layer hidden-state, (all hidden_states), (all attentions)


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(DistilBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, input_ids,  attention_mask=None, head_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]                    # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                    # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)