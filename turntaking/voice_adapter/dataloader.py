import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import WhisperProcessor
import librosa

class CommonVoiceDataset(IterableDataset):
    def __init__(self, split="train", max_samples=None, processor=None, num_shards=1, shard_id=0):
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0", 
            "en", 
            split=split, 
            trust_remote_code=True, 
            streaming=True
        )
        self.processor = processor or WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.max_samples = max_samples
        self.num_shards = num_shards
        self.shard_id = shard_id

    
    def _preprocess_text(self, text):
        """Apply your text preprocessing rules on-the-fly"""
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]  # Remove surrounding quotes
        
        if text[-1] not in [".", "?", "!"]:
            text = text + "."  # Add missing punctuation
        
        return text

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Sharding logic
        if worker_info is not None:
            total_shards = worker_info.num_workers * self.num_shards
            shard_id = worker_info.id + (self.shard_id * worker_info.num_workers)
        else:
            total_shards = self.num_shards
            shard_id = self.shard_id
        
        count = 0
        for idx, item in enumerate(self.dataset):
            # Shard the dataset
            if idx % total_shards != shard_id:
                continue
            
            if item["audio"] is not None and item["sentence"] is not None:
                try:
                    audio = item["audio"]["array"]
                    # Downsample the audio to 16000 Hz
                    audio = librosa.resample(audio, orig_sr=48000, target_sr=16000)         
                    text = self._preprocess_text(item["sentence"])  # Preprocess text here

                    inputs = self.processor(
                        audio, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    )
                    
                    yield {
                        "input_features": inputs.input_features.squeeze(0),
                        "text": text  # Use preprocessed text
                    }
                    
                    count += 1
                    if self.max_samples and count >= self.max_samples:
                        break
                except Exception as e:
                    print(f"Skipping corrupted sample: {e}")
                    continue

    def __len__(self):
        return self.max_samples if self.max_samples is not None else float('inf')
    

    
def collate_fn( batch, processor=None, tokenizer=None, device="cpu"):
    """
    Batch processing with:
    - Dynamic audio feature padding (if needed)
    - Parallel tokenization on CPU
    - Non-blocking GPU transfers
    """
    # Extract input features and texts
    input_features = [x["input_features"] for x in batch]
    texts = [x["text"] for x in batch]

    # Pad audio features if they have variable lengths (unlikely for Whisper)
    # if input_features[0].dim() == 2:  # [time, dim]
    #     input_features = pad_sequence(
    #         input_features, 
    #         batch_first=True, 
    #         padding_value=processor.feature_extractor.pad_token_id
    #     )
    # else:
    input_features = torch.stack(input_features)

    # Tokenize text in parallel (faster than sequential)
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    # Non-blocking transfer to GPU (if available)
    input_features = input_features.to(device, non_blocking=True)
    input_ids = tokenized.input_ids.to(device, non_blocking=True)
    attention_mask = tokenized.attention_mask.to(device, non_blocking=True)

    return input_features, input_ids, attention_mask