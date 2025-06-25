from transformers import WhisperForConditionalGeneration, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn
import torch
from adapter_modules import CNNAdapter, LinearAdapter, LinearAttentionAdapter

class UnifiedModelWithAdapter(nn.Module):
    def __init__(self, 
                 whisper_model, 
                 qianwen_model,
                 adapter_module=None,
                 streaming_mode=True):
        super().__init__()
        self.whisper = whisper_model
        self.adapter_module = adapter_module
        self.qianwen = qianwen_model

        
        # Verify dimensions match between model and adapter
        if adapter_module is not None:
            assert whisper_model.config.hidden_size == adapter_module.whisper_dim, \
                f"Whisper dimension mismatch: model {whisper_model.config.hidden_size} != adapter {adapter_module.whisper_dim}"
            assert qianwen_model.config.hidden_size == adapter_module.qianwen_dim, \
                f"Qianwen dimension mismatch: model {qianwen_model.config.hidden_size} != adapter {adapter_module.qianwen_dim}"

    def reset_cache(self):
        """Reset streaming cache if adapter supports it"""
        if hasattr(self.adapter_module, 'reset_cache'):
            self.adapter_module.reset_cache()

    def forward(self, audio_input, text_input=None, reset_cache=False, **kwargs):
        # Process audio through Whisper
        whisper_output = self.whisper(audio_input, output_hidden_states=True, **kwargs)
        last_hidden = whisper_output.hidden_states[-1][-1]
        
        # Process through adapter (handles dimension conversion internally)
        adapted = self.adapter_module(last_hidden, reset_cache=reset_cache)
        
        # Prepare for text generation
        if text_input is not None:
            text_embeds = self.qianwen.get_input_embeddings()(text_input)
            inputs_embeds = torch.cat([adapted, text_embeds], dim=1)
            return self.qianwen(inputs_embeds=inputs_embeds, **kwargs)
        return self.qianwen.generate(inputs_embeds=adapted, **kwargs)

def create_adapter_model(
    whisper_model_name: str = "openai/whisper-large-v3",
    qianwen_model_name: str = "Qwen/Qwen-7B",
    adapter_path: str = None,
    adapter_type: str = 'cnn',
    streaming_mode: bool = True,
    **adapter_kwargs):
    """
    Creates unified model with adapter handling dimension conversion
    
    Args:
        whisper_model_name: Whisper model name/path
        qianwen_model_name: Qianwen model name/path
        adapter_path: Path to pretrained adapter weights
        adapter_type: Type of adapter ('cnn', 'linear', 'linear_attn')
        streaming_mode: Enable streaming processing
        adapter_kwargs: Additional adapter arguments
    """
    # Load base models
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
    qianwen = AutoModelForCausalLM.from_pretrained(qianwen_model_name)
    
    # Initialize adapter
    adapter = None
    if not adapter_path:
        adapter_class = {
            'cnn': CNNAdapter,
            'linear': LinearAdapter,
            'linear_attn': LinearAttentionAdapter
        }[adapter_type]
        
        adapter_module = adapter_class(
            whisper_dim=whisper.config.hidden_size,  # Directly use model dimensions
            qianwen_dim=qianwen.config.hidden_size,
            streaming_mode=streaming_mode,
            **adapter_kwargs
        )
    else:
        # For pretrained adapters, verify dimensions match
        whisper = PeftModel.from_pretrained(whisper, adapter_path)
        whisper = whisper.merge_and_unload()
        # Assume pretrained adapter already has correct dimensions

    return UnifiedModelWithAdapter(
        whisper_model=whisper,
        adapter_module=adapter_module,
        qianwen_model=qianwen,
        streaming_mode=streaming_mode
    )