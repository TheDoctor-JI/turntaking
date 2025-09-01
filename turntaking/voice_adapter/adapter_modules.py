import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM

class BaseVoiceAdapter(nn.Module):
    def __init__(self, 
                 whisper_dim=1280,
                 qianwen_dim=4096,
                 streaming_mode=True,
                 target_seq_len=None,
                 
                 chunk_size=1600):
        super().__init__()
        self.whisper_dim = whisper_dim
        self.qianwen_dim = qianwen_dim
        self.streaming_mode = streaming_mode
        self.chunk_size = chunk_size
        
    def reset_cache(self, batch_size=None, device=None):
        """Reset cache for streaming mode (to be optionally implemented by subclasses)"""
        self.cache = None  # Default no cache
        
    def _process_chunk(self, x):
        """Process a single chunk (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def forward(self, x, reset_cache=False):
        """
        Args:
            x: (B, T, D)
            reset_cache: whether to reset streaming cache
        """
        B, T, _ = x.shape
        device = x.device

        if reset_cache:
            self.reset_cache(batch_size=B, device=device)

        if not self.streaming_mode:
            return self._process_chunk(x)

        # Streaming mode: process in chunks
        outputs = []
        for i in range(0, T, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]  # (B, C, chunk)
            processed = self._process_chunk(chunk)
            outputs.append(processed)
        return torch.cat(outputs, dim=1)


class CNNAdapter(BaseVoiceAdapter):
    def __init__(self, 
                 conv_channels=1024,
                 kernel_size=5,
                 downsample_factor=2,  # total reduction factor
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv1d(
            in_channels=self.whisper_dim, 
            out_channels=conv_channels, 
            kernel_size=kernel_size,
            stride=downsample_factor  # first downsample
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels, 
            out_channels=conv_channels * 2, 
            kernel_size=kernel_size,
            stride= 2 # optional: more downsampling if needed
        )
        self.norm = nn.LayerNorm(conv_channels * 2)
        self.proj = nn.Linear(conv_channels * 2, self.qianwen_dim)

        self.cache = None
        self.downsample_factor = downsample_factor

    def reset_cache(self, batch_size=None, device=None):
        if batch_size is not None and device is not None:
            self.cache = torch.zeros(
                batch_size, self.whisper_dim, self.kernel_size - 1, device=device
            )
        else:
            self.cache = None

    def forward(self, x, reset_cache=False):
        B, T, _ = x.shape
        device = x.device

        if reset_cache or self.cache is None:
            self.reset_cache(batch_size=B, device=device)

        if not self.streaming_mode:
            return self._process_chunk(x)

        outputs = []
        for i in range(0, T, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]
            processed = self._process_chunk(chunk)
            outputs.append(processed)
        return torch.cat(outputs, dim=1)
    
    def _process_chunk(self, x):
        """
        x: (B, T, D) -> returns (B, T', qianwen_dim)
        """
        x = x.transpose(1, 2)  # (B, D, T)

        if self.streaming_mode:
            x = torch.cat([self.cache, x], dim=2)
            self.cache = x[:, :, -self.kernel_size + 1:]

        # First conv with downsampling
        x = self.conv1(F.pad(x, (self.kernel_size - 1, 0)))  # (B, C, T//ds)
        x = F.gelu(x)

        # Second conv (optional extra downsampling)
        x = self.conv2(F.pad(x, (self.kernel_size - 1, 0)))  # (B, C2, T')
        x = x.transpose(1, 2)  # (B, T', C2)

        x = self.norm(x)
        return self.proj(x)  # (B, T', qianwen_dim)



class LinearAdapter(BaseVoiceAdapter):
    def __init__(self, hidden_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.whisper_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.qianwen_dim)
        )

    def _process_chunk(self, x):
        return self.mlp(x)


class LinearAttentionAdapter(BaseVoiceAdapter):
    def __init__(self, 
                 hidden_dim=1024,
                 attn_heads=4,
                 attn_window=8,
                 **kwargs):
        super().__init__(**kwargs)
        self.attn_window = attn_window

        # self.mlp = nn.Sequential(
        #     nn.Linear(self.whisper_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim)
        # )

        self.attn = nn.MultiheadAttention(
            self.whisper_dim, attn_heads, batch_first=True
        )

        self.proj = nn.Linear(self.whisper_dim, self.qianwen_dim)
class LinearSequenceAdapter(BaseVoiceAdapter):
    def __init__(self, target_seq_len=None, hidden_dim=512, max_speech_tokens=128, **kwargs):
        super().__init__(**kwargs)
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        self.max_speech_tokens = max_speech_tokens
        
        # Linear projection layers
        self.seq_adapter = nn.Sequential(
            nn.Linear(self.whisper_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.qianwen_dim)
        )
        
        # Learnable position embeddings for target sequence
        if target_seq_len:
            self.pos_embeds = nn.Parameter(
                torch.randn(target_seq_len, hidden_dim) * 0.02
            )
        
        # Trainable pseudo-token embeddings for speech
        self.speech_token_embeddings = nn.Parameter(
            torch.randn(max_speech_tokens, self.qianwen_dim) * 0.02
        )
        
        # Cache for streaming mode
        self.cache = None
    
    def reset_cache(self, batch_size=None, device=None):
        """Reset cache for streaming mode"""
        self.cache = None
    
    def _adapt_sequence_length(self, x, target_seq_len):
        """Adapt sequence length using interpolation"""
        batch_size, current_seq_len, hidden_dim = x.shape
        
        if current_seq_len == target_seq_len:
            return x
        
        # Transpose for interpolation: [batch, hidden, seq]
        x = x.transpose(1, 2)
        
        if current_seq_len > target_seq_len:
            # Downsample using adaptive pooling
            x = F.adaptive_avg_pool1d(x, target_seq_len)
        else:
            # Upsample using linear interpolation
            x = F.interpolate(x, size=target_seq_len, mode='linear', align_corners=False)
        
        # Transpose back: [batch, seq, hidden]
        return x.transpose(1, 2)
    
    def _process_chunk(self, x, target_seq_len=None):
        """
        Process a single chunk with sequence length adaptation
        Args:
            x: (B, T, whisper_dim)
            target_seq_len: desired output sequence length
        Returns:
            (B, target_T, qianwen_dim)
        """
        # Determine target sequence length
        if target_seq_len is None:
            target_seq_len = self.target_seq_len or x.size(1)
        
        # Adapt sequence length
        x = self._adapt_sequence_length(x, target_seq_len)
        
        # Add positional embeddings if available
        x = self.seq_adapter[0](x)  # nn.Linear(self.whisper_dim, hidden_dim)
        if hasattr(self, 'pos_embeds') and x.size(1) == self.pos_embeds.size(0):
            x = x + self.pos_embeds.unsqueeze(0)
        # Continue with the rest of seq_adapter
        x = self.seq_adapter[1:](x)
        
        # Apply sequence adapter
        return x # (B, T, qianwen_dim)
    
    def forward(self, x, reset_cache=False, target_seq_len=None, labels=None):
        """
        Forward pass with optional sequence length matching
        Args:
            x: (B, T, whisper_dim)
            reset_cache: whether to reset streaming cache
            target_seq_len: desired output sequence length
            labels: if provided, use labels.size(1) as target length
        """
        B, T, _ = x.shape
        device = x.device
        
        if reset_cache:
            self.reset_cache(batch_size=B, device=device)
        
        # Determine target sequence length from labels if provided
        if labels is not None:
            target_seq_len = labels.size(1)
        elif target_seq_len is None:
            target_seq_len = self.target_seq_len
        
        # If still None, keep original length
        if target_seq_len is None:
            target_seq_len = T
        
        if not self.streaming_mode:
            proj = self._process_chunk(x, target_seq_len)  # (B, T, qianwen_dim)
        else:
            # Streaming mode: process in chunks
            outputs = []
            for i in range(0, T, self.chunk_size):
                chunk = x[:, i:i+self.chunk_size]
                chunk_target_len = None
                if target_seq_len:
                    chunk_target_len = min(target_seq_len, 
                                         int(target_seq_len * chunk.size(1) / T))
                processed = self._process_chunk(chunk, chunk_target_len)
                outputs.append(processed)
            
            proj = torch.cat(outputs, dim=1)
            
            # Final sequence length adaptation if needed
            if target_seq_len and proj.size(1) != target_seq_len:
                proj = self._adapt_sequence_length(proj, target_seq_len)

        # ---- NEW STEP: Map into speech-token embedding subspace ----
        # logits: (B, T, qianwen_dim)
        # speech_token_embeddings: (num_tokens, qianwen_dim)
        weights = torch.matmul(proj, self.speech_token_embeddings.T)  # (B, T, num_tokens)
        weights = F.softmax(weights, dim=-1)
        proj = torch.matmul(weights, self.speech_token_embeddings)  # (B, T, qianwen_dim)

        return proj



class LinearAttentionSequenceAdapter(BaseVoiceAdapter):
    def __init__(self, target_seq_len=512, hidden_dim=512, attn_heads=8,labels=None, max_speech_tokens=128, **kwargs):
        super().__init__(**kwargs)
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        
        # Input projection
        self.input_proj = nn.Linear(self.whisper_dim, hidden_dim)
        
        # Learnable query tokens for target sequence length
        self.query_tokens = nn.Parameter(
            torch.randn(target_seq_len, hidden_dim) * 0.02
        )
        # Trainable pseudo-token embeddings for speech
        self.speech_token_embeddings = nn.Parameter(
            torch.randn(max_speech_tokens, self.qianwen_dim) * 0.02
        )
        # Cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            batch_first=True
        )
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.qianwen_dim)
        
        # Cache for streaming mode
        self.cache = None
    
    def reset_cache(self, batch_size=None, device=None):
        """Reset cache for streaming mode"""
        self.cache = None
    
    def _process_chunk(self, x, target_seq_len=None):
        """
        Process a single chunk using cross-attention
        Args:
            x: (B, T, whisper_dim)
            target_seq_len: desired output sequence length
        Returns:
            (B, target_seq_len, qianwen_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        key_value = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Determine target sequence length
        if target_seq_len is None:
            target_seq_len = self.target_seq_len
        
        # Prepare query tokens
        if target_seq_len <= self.query_tokens.size(0):
            queries = self.query_tokens[:target_seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # If target is longer than prepared queries, repeat the pattern
            queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            repeats = (target_seq_len + self.query_tokens.size(0) - 1) // self.query_tokens.size(0)
            queries = queries.repeat(1, repeats, 1)[:, :target_seq_len, :]
        
        # Cross-attention: queries attend to audio features
        attended, attention_weights = self.cross_attention(
            query=queries,
            key=key_value,
            value=key_value
        )
        
        # Residual connection and layer norm
        attended = self.norm1(attended + queries)
        
        # Feed forward network
        ffn_output = self.ffn(attended)
        output = self.norm2(ffn_output + attended)
        
        # Project to qianwen dimension
        return self.output_proj(output)
    
    def forward(self, x, reset_cache=False, target_seq_len=None, labels=None):
        """
        Forward pass with cross-attention sequence adaptation
        Args:
            x: (B, T, whisper_dim)
            reset_cache: whether to reset streaming cache
            target_seq_len: desired output sequence length
            labels: if provided, use labels.size(1) as target length
        """
        B, T, _ = x.shape
        device = x.device

        if reset_cache:
            self.reset_cache(batch_size=B, device=device)
        
        # Determine target sequence length from labels if provided
        if labels is not None:
            target_seq_len = labels.size(1)
        elif target_seq_len is None:
            target_seq_len = self.target_seq_len
        
        if not self.streaming_mode:
            print(x.shape, target_seq_len)
            return self._process_chunk(x, target_seq_len)

        # Streaming mode: process in chunks
        outputs = []
        chunk_target_len = max(1, target_seq_len // ((T + self.chunk_size - 1) // self.chunk_size))
        
        for i in range(0, T, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]
            processed = self._process_chunk(chunk, chunk_target_len)
            outputs.append(processed)
        
        result = torch.cat(outputs, dim=1)
        
        # Final length adjustment if needed
        if result.size(1) != target_seq_len:
            # Use interpolation for final adjustment
            result = result.transpose(1, 2)  # (B, qianwen_dim, seq)
            result = F.interpolate(result, size=target_seq_len, mode='linear', align_corners=False)
            proj = result.transpose(1, 2)  # (B, target_seq_len, qianwen_dim)
        # ---- NEW STEP: Map into speech-token embedding subspace ----

        # logits: (B, T, qianwen_dim)
        # speech_token_embeddings: (num_tokens, qianwen_dim)
        weights = torch.matmul(proj, self.speech_token_embeddings.T)  # (B, T, num_tokens)
        weights = F.softmax(weights, dim=-1)
        proj = torch.matmul(weights, self.speech_token_embeddings)  # (B, T, qianwen_dim)
        return proj
    # def _process_chunk(self, x):
    #     x = self.mlp(x)  # (B, T, D)

    #     if self.streaming_mode:
    #         attn_mask = self._create_sliding_mask(x.size(1), x.device)
    #         x, _ = self.attn(x, x, x, attn_mask=attn_mask)
    #     else:
    #         x, _ = self.attn(x, x, x)

    #     return self.proj(x)

    # def _create_sliding_mask(self, seq_len, device):
    #     # Lower-triangular with limited window
    #     mask = torch.ones(seq_len, seq_len, device=device)
    #     for i in range(seq_len):
    #         mask[i, max(0, i - self.attn_window):i + 1] = 0
    #     return mask.bool()


def run_adapter_forward(encoder_output, adapter: BaseVoiceAdapter, streaming=False):
    """
    Run forward pass with an adapter
    Args:
        encoder_output: Tensor (B, T, D)
        adapter: instance of BaseVoiceAdapter
        streaming: bool, whether to use streaming
    """
    adapter.streaming_mode = streaming
    return adapter(encoder_output, reset_cache=True)
# # Modified processing function to use adapter factory
# def process_audio_with_adapter(audio_path, 
#                              adapter_type='cnn',
#                              streaming_mode=True,
#                              qianwen_model_name="Qwen/Qwen-7B"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load models
#     whisper_processor, whisper_model = load_whisper_v3()
#     whisper_model = whisper_model.to(device).eval()
    
#     qianwen_tokenizer = AutoTokenizer.from_pretrained(qianwen_model_name)
#     qianwen_model = AutoModelForCausalLM.from_pretrained(
#         qianwen_model_name,
#         torch_dtype=torch.float16 if device == "cuda" else torch.float32
#     ).to(device).eval()
    
#     # Initialize selected adapter
#     adapter = create_adapter(
#         adapter_type=adapter_type,
#         streaming_mode=streaming_mode
#     ).to(device)
    
#     # Process audio
#     with torch.no_grad():
#         input_features = whisper_processor(
#             audio_path, return_tensors="pt", sampling_rate=16000
#         ).input_features.to(device)
        
#         outputs = whisper_model.generate(
#             input_features,
#             output_hidden_states=True,
#             return_dict_in_generate=True,
#             max_new_tokens=1
#         )
#         last_hidden = outputs.hidden_states[-1][-1]
        
#         adapted = adapter(last_hidden.unsqueeze(0))
        
#         prompt = "Based on the audio input:"
#         prompt_tokens = qianwen_tokenizer(prompt, return_tensors="pt").to(device)
#         prompt_embeds = qianwen_model.get_input_embeddings()(prompt_tokens.input_ids)
        
#         inputs_embeds = torch.cat([prompt_embeds, adapted], dim=1)
        
#         outputs = qianwen_model.generate(
#             inputs_embeds=inputs_embeds,
#             max_new_tokens=200,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7
#         )
        
#     return qianwen_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# # Example usage
# if __name__ == "__main__":
#     audio_path = "example_audio.mp3"
    
#     # Try different configurations
#     for adapter_type in ['cnn', 'linear', 'linear_attn']:
#         print(f"\nUsing {adapter_type} adapter:")
#         result = process_audio_with_adapter(
#             audio_path,
#             adapter_type=adapter_type,
#             streaming_mode=True
#         )
#         print("Response:", result[:200] + "...")  # Print first 200 chars