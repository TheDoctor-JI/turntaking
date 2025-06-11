import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataloader import CommonVoiceDataset, collate_fn
from your_adapter_file import create_adapter

from jiwer import wer
from torchmetrics.text import WordErrorRate, CharErrorRate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
from huggingface_hub import HfApi, Repository
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ✅ WandB project initialization
wandb.init(project="voice-adapter-training", config={
    "adapter_type": "cnn",
    "epochs": 20,
    "batch_size": 4,
    "lr": 1e-4,
    "freeze_threshold_wer": 0.25,
    "unfreeze_after_wer": 0.25,
    "patience": 3
})
config = wandb.config


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_last_whisper_layers(whisper_model, num_layers=4):
    for layer in whisper_model.model.encoder.layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def train_adapter(
    whisper_name="openai/whisper-large-v3",
    qianwen_name="Qwen/Qwen-7B",
    adapter_type="cnn",
    max_epochs=10,
    batch_size=4,
    lr=1e-4,
    max_samples=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    freeze_threshold_wer=0.25
):
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_name).to(device)
    processor = WhisperProcessor.from_pretrained(whisper_name)
    tokenizer = AutoTokenizer.from_pretrained(qianwen_name)
    freeze_model(whisper_model)

    adapter = create_adapter(adapter_type=adapter_type).to(device)

    output_dim = processor.tokenizer.vocab_size
    head = nn.Linear(adapter.qianwen_dim, output_dim).to(device)

    ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=lr
    )


    full_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, processor, tokenizer, device))
    for epoch in range(max_epochs):
        adapter.train()
        whisper_model.eval()
        total_loss = 0.0

        references = []
        hypotheses = []

        for input_features, labels, attention_mask in loader:
            with torch.no_grad():
                outputs = whisper_model(input_features, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states[-1]

            adapted = adapter(hidden_states)
            logits = head(adapted)  # (B, T, V)
            log_probs = logits.log_softmax(dim=-1)

            input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long).to(device)
            target_lengths = attention_mask.sum(dim=1).to(device)

            loss = ctc_loss(
                log_probs.transpose(0, 1),  # CTC expects (T, B, V)
                labels,
                input_lengths,
                target_lengths
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Decode predictions and accumulate references/hypotheses for metrics
            predicted_ids = log_probs.argmax(-1)
            for pred, label in zip(predicted_ids, labels):
                pred_str = processor.tokenizer.decode(pred, skip_special_tokens=True)
                label_str = tokenizer.decode(label, skip_special_tokens=True)
                hypotheses.append(pred_str)
                references.append(label_str)

        avg_loss = total_loss / len(loader)
        current_wer = wer(references, hypotheses)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | WER: {current_wer:.4f}")

        # Evaluation metrics
        all_preds_flat = " ".join(hypotheses).split()
        all_refs_flat = " ".join(references).split()

        def to_binary(a, b):
            return [1 if x in b else 0 for x in a], [1]*len(a)

        y_pred_bin, y_true_bin = to_binary(all_preds_flat, all_refs_flat)

        print(f"Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.4f}")
        print(f"Precision: {precision_score(y_true_bin, y_pred_bin, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true_bin, y_pred_bin, zero_division=0):.4f}")
        print(f"F1 Score: {f1_score(y_true_bin, y_pred_bin, zero_division=0):.4f}")

        # Trigger unfreezing of Whisper layers if threshold met
        if current_wer < freeze_threshold_wer:
            print("Threshold met. Unfreezing last 4 Whisper layers...")
            unfreeze_last_whisper_layers(whisper_model, 4)
            whisper_model.train()
            optimizer.add_param_group({
                'params': [p for p in whisper_model.parameters() if p.requires_grad],
                'lr': lr / 10  # Smaller LR for Whisper fine-tuning
            })

    torch.save(adapter.state_dict(), f"{adapter_type}_adapter_final.pt")
    print("Training complete.")
