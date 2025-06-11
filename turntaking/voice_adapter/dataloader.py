import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor

class CommonVoiceDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        self.dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split=split)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.dataset = self.dataset.filter(lambda x: x["audio"] is not None and x["text"] is not None)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        text = item["text"]
        
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        return {
            "input_features": inputs.input_features.squeeze(0),
            "text": text
        }

def collate_fn(batch, processor, tokenizer, device="cpu"):
    input_features = torch.stack([x["input_features"] for x in batch]).to(device)
    text = [x["text"] for x in batch]
    tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    return input_features, tokenized.input_ids, tokenized.attention_mask
