from transformers import  WhisperModel, AutoModelForCausalLM
from peft import PeftModel
import torch.nn as nn
import torch
from adapter_modules import CNNAdapter, LinearAdapter, LinearAttentionAdapter, LinearSequenceAdapter, LinearAttentionSequenceAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM

class UnifiedModelWithAdapter(nn.Module):
    def __init__(self, 
                 whisper_model, 
                 qianwen_model,
                 adapter_module=None,
                 tokenizer=None,
                 streaming_mode=True):
        super().__init__()
        self.whisper = whisper_model
        self.adapter_module = adapter_module
        self.qianwen = qianwen_model
        self.tokenizer = tokenizer

        # # Create learned timestamp embeddings
        # self.timestamp_embeds = nn.Embedding(100, qianwen_model.config.hidden_size)  # 0-10 seconds in 0.1s intervals
        # nn.init.normal_(self.timestamp_embeds.weight, std=0.02)
        
        # Verify dimensions match between model and adapter
        if adapter_module is not None:
            assert whisper_model.config.hidden_size == adapter_module.whisper_dim, \
                f"Whisper dimension mismatch: model {whisper_model.config.hidden_size} != adapter {adapter_module.whisper_dim}"
            assert qianwen_model.config.hidden_size == adapter_module.qianwen_dim, \
                f"Qianwen dimension mismatch: model {qianwen_model.config.hidden_size} != adapter {adapter_module.qianwen_dim}"
        self.max_prefix_len = 128  # Adjust this as needed
        self.learned_prefix = nn.Parameter(
            torch.randn(self.max_prefix_len, qianwen_model.config.hidden_size) * 0.02
        )
    def reset_cache(self):
        """Reset streaming cache if adapter supports it"""
        if hasattr(self.adapter_module, 'reset_cache'):
            self.adapter_module.reset_cache()


    def forward(self, audio_input, attention_mask=None, labels=None, reset_cache=False, **kwargs):
        # Process audio through Whisper
        # print(f"Audio input shape: {audio_input.shape}")

        # whisper_output = self.whisper(audio_input, attention_mask=attention_mask, language='en', **kwargs)
        

    
        # decoder_inputs_embeds = self.whisper.get_input_embeddings()(torch.tensor(decoder_input_ids, device=audio_input.device)).unsqueeze(0)
        # whisper_output = self.whisper(audio_input, decoder_inputs_embeds=decoder_inputs_embeds)
        # whisper_output = whisper_output.last_hidden_state
        encoder_outputs = self.whisper(audio_input)
        whisper_output = encoder_outputs.last_hidden_state
        
        # Create longer decoder input sequence for teacher forcing
        # batch_size = audio_input.size(0)
        
 
        
        # print(f"Whisper output shape: {whisper_output.shape}")
            
        # breakpoint()
        # last_hidden = whisper_output.hidden_states[-1][-1]
        # print(f"Whisper output shape: {last_hidden.shape}")
        # breakpoint()
        # Process through adapter (handles dimension conversion internally)
        adapted = self.adapter_module(whisper_output, reset_cache=reset_cache, labels=labels)
        # print(f"Adapted audio shape: {adapted.shape}")
  
        # If adapted is already token ids:
      # batch_size = adapted.size(0)
        # seq_len = adapted.size(1)
        # timestamp_indices = torch.arange(0, min(seq_len, 100), device=adapted.device)
        # timestamp_indices = timestamp_indices.unsqueeze(0).expand(batch_size, -1)
        # Prepend audio token id

        
        # Verify tokens were added
        audio_token_id = self.tokenizer.convert_tokens_to_ids("</audio>")

  
        
        # # Get timestamp embeddings
        # timestamp_embeds = self.timestamp_embeds(timestamp_indices[:, :seq_len])
        
        # # 4. Combine audio features with timestamp info
        # combined_features = adapted + timestamp_embeds  # Or concatenate
        
        audio_token_embed = self.qianwen.get_input_embeddings()(
            torch.tensor([audio_token_id], device=adapted.device)
        )  # (1, embed_dim)
        audio_token_embed = audio_token_embed.unsqueeze(0).expand(adapted.size(0), -1, -1)  # (batch, 1, embed_dim)
        input_embeds = torch.cat([audio_token_embed, adapted], dim=1)  # (batch, seq_len+1)

        # Convert all token ids to embeddings

        # print(f"Input embeddings shape after prepending audio token: {input_embeds.shape}")
        outputs=self.qianwen(inputs_embeds=input_embeds, **kwargs)
        logits=outputs.logits
        # print(f"Logits shape: {logits.shape}")
        return logits

def create_adapter_model(
    whisper_model_name: str = "openai/whisper-large-v3",
    qianwen_model_name: str = "Qwen/Qwen-7B",
    adapter_path: str = None,
    adapter_type: str = 'cnn',
    streaming_mode: bool = True,
    target_seq_len: int = None,  # Add this parameter
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

    tokenizer = AutoTokenizer.from_pretrained(qianwen_model_name,
        trust_remote_code=True,
    )
    special_tokens_dict = {"additional_special_tokens": ["</audio>"]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added} special tokens to tokenizer: {special_tokens_dict['additional_special_tokens']}")
    print(f"Tokenizer size after adding special tokens: {len(tokenizer)}")
    # tokenizer.pad_token_id = tokenizer.pad_token_id 
    # whisper_tokens=[f"<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    # special_tokens_dict["additional_special_tokens"].extend(whisper_tokens)
    # timestamp_tokens = [f"<|{i/50:.2f}|>" for i in range(0, 1500)]  # 0.00 to 30.00 seconds
    # special_tokens_dict["additional_special_tokens"].extend(timestamp_tokens)
    
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens to tokenizer")

    # If you are using a model, resize its embeddings to match the new tokenizer size
    # print(len(qianwen.get_input_embeddings().weight))
    # Load base models
    whisper = WhisperModel.from_pretrained(whisper_model_name)
    whisper=whisper.encoder
    # Freeze all parameters
# Freeze all parameters
    for param in whisper.parameters():
        param.requires_grad = False
    # # Unfreeze only the last hidden layer of the decoder
    # num_decoder_layers = len(whisper.decoder.layers)
    # num_layers_to_unfreeze = 12
    # start_layer = max(0, num_decoder_layers - num_layers_to_unfreeze)

    # for layer_idx, layer in enumerate(whisper.decoder.layers):
    #     if layer_idx >= start_layer:
    #         for param in layer.parameters():
    #             param.requires_grad = True

    # Verify
    # for name, param in whisper.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")
    qianwen = AutoModelForCausalLM.from_pretrained(qianwen_model_name)
    qianwen.resize_token_embeddings(len(tokenizer))
    print(f"Model resized to match tokenizer size: {len(qianwen.get_input_embeddings().weight)}")
    print("Model embedding size:", qianwen.get_input_embeddings().weight.shape[0])

    # Initialize adapter
    adapter = None
    if not adapter_path:
        adapter_class = {
            'cnn': CNNAdapter,
            'linear': LinearAdapter,
            'linear_attn': LinearAttentionAdapter,
            'linear_seq': LinearSequenceAdapter,           # New
            'linear_attn_seq': LinearAttentionSequenceAdapter,  # New
        }[adapter_type]
    
        adapter_module = adapter_class(
            whisper_dim=whisper.config.hidden_size,  # Directly use model dimensions
            qianwen_dim=1024,
            streaming_mode=streaming_mode,
            target_seq_len=target_seq_len,  # Pass target length
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
        tokenizer=tokenizer,
        streaming_mode=streaming_mode
    )