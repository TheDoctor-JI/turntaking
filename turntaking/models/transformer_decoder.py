import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from turntaking.models.multi_head_attention import (
    MultiHeadAttentionAlibi,
    MultiHeadAttention,
)
from turntaking.models.transformer import StaticPositionEmbedding, ffn_block


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with cross-attention (first layer) or self-attention (subsequent layers)
    """

    def __init__(
        self,
        dim: int = 512,
        ffn_dim: int = 1536,
        num_heads: int = 8,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        position_emb: bool = False,
        use_pre_ln: bool = True,
        is_first_layer: bool = False,
    ):
        super().__init__()
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_cross_attn = nn.LayerNorm(dim) if is_first_layer else None
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout)
        self.use_pre_ln = use_pre_ln
        self.is_first_layer = is_first_layer

        # Attention modules
        if position_emb:
            self.self_attention = MultiHeadAttention(
                dim=dim, num_heads=num_heads, dropout=dropout
            )
            if is_first_layer:
                self.cross_attention = MultiHeadAttention(
                    dim=dim, num_heads=num_heads, dropout=dropout
                )
        else:
            self.self_attention = MultiHeadAttentionAlibi(
                dim=dim, num_heads=num_heads, dropout=dropout
            )
            if is_first_layer:
                self.cross_attention = MultiHeadAttentionAlibi(
                    dim=dim, num_heads=num_heads, dropout=dropout
                )
        
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (decoder representations)
            encoder_output: Encoder output (only used in first layer for cross-attention)
            mask: Attention mask
        """
        if self.use_pre_ln:
            if self.is_first_layer and encoder_output is not None:
                # First layer: Cross-attention
                x_norm = self.ln_cross_attn(x)
                enc_norm = self.ln_cross_attn(encoder_output)
                cross_attn_out, attn = self.cross_attention(Q=x_norm, K=enc_norm, V=enc_norm, mask=mask)
                x = x + self.dropout(cross_attn_out)
            else:
                # Self-attention (for subsequent layers or when no encoder output)
                x_norm = self.ln_self_attn(x)
                self_attn_out, attn = self.self_attention(Q=x_norm, K=x_norm, V=x_norm, mask=mask)
                x = x + self.dropout(self_attn_out)
            
            # Feed-forward network
            x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        else:
            # Post-layer normalization
            if self.is_first_layer and encoder_output is not None:
                cross_attn_out, attn = self.cross_attention(Q=x, K=encoder_output, V=encoder_output, mask=mask)
                x = self.ln_cross_attn(x + cross_attn_out)
            else:
                self_attn_out, attn = self.self_attention(Q=x, K=x, V=x, mask=mask)
                x = self.ln_self_attn(x + self_attn_out)
            
            x = self.ln_ffnetwork(x + self.ffnetwork(x))
        
        return x, attn


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with cross-attention in first layer and self-attention in subsequent layers
    """
    
    def __init__(
        self,
        input_size: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
        use_pos_emb: bool = False,
        max_context: int = 1024,
        use_pre_ln: bool = True,
    ):
        super().__init__()
        self.dim = input_size
        self.dff = int(input_size * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout
        self.use_pos_emb = use_pos_emb
        self.use_pre_ln = use_pre_ln

        if self.use_pos_emb:
            self.max_context = max_context
            self.pos_emb = StaticPositionEmbedding(max_context, self.dim)
        else:
            self.pos_emb = nn.Identity()

        layers = []
        for i in range(self.num_layers):
            layers.append(
                TransformerDecoderLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    position_emb=self.use_pos_emb,
                    use_pre_ln=self.use_pre_ln,
                    is_first_layer=(i == 0),  # Only first layer does cross-attention
                )
            )
        self.layers = nn.ModuleList(layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, decoder_input, encoder_output, attention=False):
        """
        Args:
            decoder_input: Decoder input tensor (wc - waveform conditioning) [B, T, D]
            encoder_output: Encoder output tensor (vc - VAD conditioning) [B, T, D]
        """
        all_attention = []

        x = self.pos_emb(decoder_input)
        encoder_output = self.pos_emb(encoder_output)

        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer: cross-attention with encoder output
                x, attn = layer(x, encoder_output)
            else:
                # Subsequent layers: self-attention
                x, attn = layer(x)
            
            if attention:
                all_attention.append(attn)

        if attention:
            attn = torch.stack(all_attention, dim=1)
            return x, attn

        return x


def _test_transformer_decoder():
    model = TransformerDecoder(input_size=256, dff_k=3, num_layers=3, num_heads=4)
    
    B, T, D = 4, 100, 256
    decoder_input = torch.rand(B, T, D)  # waveform conditioning
    encoder_output = torch.rand(B, T, D)  # VAD conditioning
    
    with torch.no_grad():
        output = model(decoder_input, encoder_output)
        print("Output shape:", output.shape)
        
        output_with_attn, attn = model(decoder_input, encoder_output, attention=True)
        print("Output with attention shape:", output_with_attn.shape)
        print("Attention shape:", attn.shape)


if __name__ == "__main__":
    _test_transformer_decoder()