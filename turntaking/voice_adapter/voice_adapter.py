import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseVoiceAdapter(nn.Module):
    def __init__(self, 
                 whisper_dim=1280,
                 qianwen_dim=4096,
                 streaming_mode=True,
                 chunk_size=1600):
        super().__init__()
        self.whisper_dim = whisper_dim
        self.qianwen_dim = qianwen_dim
        self.streaming_mode = streaming_mode
        self.chunk_size = chunk_size
        
    def reset_cache(self):
        """Reset cache for streaming mode"""
        pass
        
    def _process_chunk(self, x):
        """Process a single chunk (to be implemented by subclasses)"""
        raise NotImplementedError
        
    def forward(self, x, reset_cache=False):
        """Base forward pass handles streaming logic"""
        if reset_cache:
            self.reset_cache()
            
        if not self.streaming_mode:
            return self._process_chunk(x)
            
        # Streaming mode processing
        outputs = []
        for i in range(0, x.size(1), self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]
            outputs.append(self._process_chunk(chunk))
        return torch.cat(outputs, dim=1)

class CNNAdapter(BaseVoiceAdapter):
    def __init__(self, 
                 conv_channels=1024,
                 kernel_size=5,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.register_buffer("cache", torch.zeros(1, self.whisper_dim, kernel_size-1))
        
        self.conv1 = nn.Conv1d(self.whisper_dim, conv_channels, kernel_size)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels*2, kernel_size)
        self.norm = nn.LayerNorm(conv_channels*2)
        self.proj = nn.Linear(conv_channels*2, self.qianwen_dim)

    def reset_cache(self):
        self.cache = torch.zeros_like(self.cache)
        
    def _process_chunk(self, x):
        """Process chunk with CNN layers"""
        x = x.transpose(1, 2)
        padded = torch.cat([self.cache, x], dim=2) if self.streaming_mode else x
        if self.streaming_mode:
            self.cache = padded[:, :, -(self.kernel_size-1):]
            
        x = self.conv1(F.pad(padded, (self.kernel_size-1, 0)))
        x = F.gelu(x)
        x = self.conv2(F.pad(x, (self.kernel_size-1, 0)))
        x = x.transpose(1, 2)
        x = self.norm(x)
        return self.proj(x)

class LinearAdapter(BaseVoiceAdapter):
    def __init__(self, hidden_dim=2048, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(self.whisper_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.qianwen_dim)
            
    def _process_chunk(self, x):
        return self.mlp(x)

class LinearAttentionAdapter(BaseVoiceAdapter):
    def __init__(self, 
                 hidden_dim=2048,
                 attn_heads=4,
                 attn_window=8,
                 **kwargs):
        super().__init__(**kwargs)
        self.attn_window = attn_window
        self.mlp = nn.Sequential(
            nn.Linear(self.whisper_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim))
            
        self.proj = nn.Linear(hidden_dim, self.qianwen_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, attn_heads, batch_first=True)

    def _process_chunk(self, x):
        x = self.mlp(x)
        if self.streaming_mode:
            attn_mask = self._create_sliding_mask(x.size(1))
            x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        else:
            x, _ = self.attn(x, x, x)
        return self.proj(x)
        
    def _create_sliding_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            mask[i, max(0, i-self.attn_window):i+1] = 0
        return mask.bool().to(next(self.parameters()).device)

def create_adapter(adapter_type='cnn', **kwargs):
    """Factory function to create adapter based on type"""
    adapters = {
        'cnn': CNNAdapter,
        'linear': LinearAdapter,
        'linear_attn': LinearAttentionAdapter
    }
    return adapters[adapter_type](**kwargs)

# Modified processing function to use adapter factory
def process_audio_with_adapter(audio_path, 
                             adapter_type='cnn',
                             streaming_mode=True,
                             qianwen_model_name="Qwen/Qwen-7B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    whisper_processor, whisper_model = load_whisper_v3()
    whisper_model = whisper_model.to(device).eval()
    
    qianwen_tokenizer = AutoTokenizer.from_pretrained(qianwen_model_name)
    qianwen_model = AutoModelForCausalLM.from_pretrained(
        qianwen_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    
    # Initialize selected adapter
    adapter = create_adapter(
        adapter_type=adapter_type,
        streaming_mode=streaming_mode
    ).to(device)
    
    # Process audio
    with torch.no_grad():
        input_features = whisper_processor(
            audio_path, return_tensors="pt", sampling_rate=16000
        ).input_features.to(device)
        
        outputs = whisper_model.generate(
            input_features,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )
        last_hidden = outputs.hidden_states[-1][-1]
        
        adapted = adapter(last_hidden.unsqueeze(0))
        
        prompt = "Based on the audio input:"
        prompt_tokens = qianwen_tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = qianwen_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        inputs_embeds = torch.cat([prompt_embeds, adapted], dim=1)
        
        outputs = qianwen_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
    return qianwen_tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# Example usage
if __name__ == "__main__":
    audio_path = "example_audio.mp3"
    
    # Try different configurations
    for adapter_type in ['cnn', 'linear', 'linear_attn']:
        print(f"\nUsing {adapter_type} adapter:")
        result = process_audio_with_adapter(
            audio_path,
            adapter_type=adapter_type,
            streaming_mode=True
        )
        print("Response:", result[:200] + "...")  # Print first 200 chars