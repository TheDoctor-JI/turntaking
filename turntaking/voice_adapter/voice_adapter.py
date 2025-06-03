import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F

class WhisperV3QianwenAdapter(nn.Module):
    """
    Enhanced adapter for Whisper Large v3 to Qianwen connection.
    Handles Whisper v3's specific architecture (1280-dim hidden states)
    with improved feature transformation.
    """
    def __init__(self, whisper_hidden_size=1280, qianwen_hidden_size=4096):
        super().__init__()
        
        # Dimensionality transformation with residual connection
        self.projection = nn.Sequential(
            nn.Linear(whisper_hidden_size, qianwen_hidden_size * 2),
            nn.GELU(),
            nn.Linear(qianwen_hidden_size * 2, qianwen_hidden_size)
        )
        
        # Feature refinement
        self.norm = nn.LayerNorm(qianwen_hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=qianwen_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Context gating mechanism
        self.context_gate = nn.Sequential(
            nn.Linear(qianwen_hidden_size, qianwen_hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, whisper_features):
        # Project features
        projected = self.projection(whisper_features)
        
        # Apply attention
        attended, _ = self.attention(projected, projected, projected)
        
        # Context gating
        gate = self.context_gate(attended)
        gated_features = attended * gate
        
        # Final normalization
        return self.norm(gated_features)

def load_whisper_v3():
    """Load Whisper Large v3 with proper configuration"""
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.config.output_hidden_states = True
    return processor, model

def process_audio_with_adapter(audio_path, qianwen_model_name="Qwen/Qwen-7B"):
    """Complete processing pipeline for Whisper v3 + Qianwen"""
    # Load models with automatic device placement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper v3
    whisper_processor, whisper_model = load_whisper_v3()
    whisper_model = whisper_model.to(device).eval()
    
    # Load Qianwen
    qianwen_tokenizer = AutoTokenizer.from_pretrained(qianwen_model_name)
    qianwen_model = AutoModelForCausalLM.from_pretrained(
        qianwen_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()
    
    # Initialize adapter
    adapter = WhisperV3QianwenAdapter().to(device)
    
    # Process audio
    with torch.no_grad():
        # Load and preprocess audio
        input_features = whisper_processor(
            audio_path,
            return_tensors="pt",
            sampling_rate=16000
        ).input_features.to(device)
        
        # Get Whisper features
        outputs = whisper_model.generate(
            input_features,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=1  # We just want the features
        )
        last_hidden = outputs.hidden_states[-1][-1]  # Get last layer of last decoder step
        
        # Adapt features
        adapted = adapter(last_hidden.unsqueeze(0))  # Add batch dimension
        
        # Prepare Qianwen input
        prompt = "Based on the audio input:"
        prompt_tokens = qianwen_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)
        
        # Get prompt embeddings
        prompt_embeds = qianwen_model.get_input_embeddings()(prompt_tokens.input_ids)
        
        # Combine with adapted features
        inputs_embeds = torch.cat([prompt_embeds, adapted], dim=1)
        
        # Generate response
        outputs = qianwen_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=qianwen_tokenizer.eos_token_id
        )
        
    # Decode and clean output
    full_text = qianwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()  # Extract just the response
    
    return response

# Example usage with memory management
if __name__ == "__main__":
    audio_path = "example_audio.mp3"
    
    try:
        with torch.inference_mode():
            result = process_audio_with_adapter(audio_path)
            print("\nGenerated Response:", result)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Out of memory! Try with smaller models or batch size")
        else:
            raise e