import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataloader import CommonVoiceDataset, collate_fn
from voice_adapter import create_adapter

from jiwer import wer
from torchmetrics.text import WordErrorRate, CharErrorRate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
from huggingface_hub import HfApi, Repository
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import atexit
import warnings



def setup(rank, world_size):
    """Initialize the distributed environment."""
    # torchrun handles MASTER_ADDR and MASTER_PORT automatically
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Register cleanup to be called on normal program termination
    atexit.register(cleanup)

def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def freeze_whisper_layers(whisper_model, num_layers_to_freeze=None):
    """Freeze all or specific layers of Whisper model"""
    # Freeze the entire model by default
    for param in whisper_model.parameters():
        param.requires_grad = False
    
    # If specific number of layers to freeze is given
    if num_layers_to_freeze is not None:
        # Unfreeze the remaining layers
        for layer in whisper_model.model.encoder.layers[num_layers_to_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

def unfreeze_whisper_layers(whisper_model, num_layers_to_unfreeze=4):
    """Unfreeze last N layers of Whisper model"""
    # Unfreeze last N encoder layers
    for layer in whisper_model.model.encoder.layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

def freeze_qwen_layers(qwen_model, num_layers_to_freeze=None):
    """Freeze all or specific layers of Qwen model"""
    # Freeze the entire model by default
    for param in qwen_model.parameters():
        param.requires_grad = False
    
    # If specific number of layers to freeze is given
    if num_layers_to_freeze is not None:
        # Unfreeze the remaining layers
        for layer in qwen_model.transformer.h[num_layers_to_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

def unfreeze_qwen_layers(qwen_model, num_layers_to_unfreeze=4):
    """Unfreeze last N layers of Qwen model"""
    # Unfreeze last N transformer layers
    for layer in qwen_model.transformer.h[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

def train_adapter_ddp(rank, world_size, args):
    try:
        setup(rank, world_size)
        
        # Set device
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        # Initialize WandB only on rank 0
        if rank == 0:
            wandb.init(project="voice-adapter-training", config=vars(args))
            config = wandb.config
        else:
            config = args

        # Load models
        whisper_model = WhisperForConditionalGeneration.from_pretrained(args.whisper_name).to(device)
        processor = WhisperProcessor.from_pretrained(args.whisper_name)

        qwen_model = AutoModelForCausalLM.from_pretrained(args.qianwen_name,trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.qianwen_name,trust_remote_code=True)

        # Freeze models initially
        freeze_whisper_layers(whisper_model)
        freeze_qwen_layers(qwen_model)

        # Create adapter and wrap with DDP
        adapter = create_adapter(adapter_type=args.adapter_type).to(device)
        adapter = DDP(adapter, device_ids=[rank])

        output_dim = processor.tokenizer.vocab_size
        head = nn.Linear(adapter.module.qianwen_dim, output_dim).to(device)

        # Initialize datasets
        train_dataset = CommonVoiceDataset(split="train", max_samples=args.max_samples)
        val_dataset = CommonVoiceDataset(split="validation", max_samples=args.max_samples//10)

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        # Load dataset with distributed sampler
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, processor, tokenizer, device),
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, processor, tokenizer, device),
            num_workers=2,
            pin_memory=True
        )

        # Loss and optimizer
        ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
        optimizer = torch.optim.AdamW(
            list(adapter.parameters()) + list(head.parameters()), 
            lr=args.lr
        )

        # Training loop
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            adapter.train()
            whisper_model.eval()
            qwen_model.eval()
            
            epoch_loss = 0.0
            references = []
            hypotheses = []

            for batch in train_loader:
                input_features, labels, attention_mask = batch

                # Reset cache at start of each batch if streaming
                if args.streaming:

                    adapter.module.reset_cache()                
                with torch.no_grad():
                    whisper_outputs = whisper_model(
                        input_features, 
                        output_hidden_states=True, 
                        return_dict=True
                    )
                    hidden_states = whisper_outputs.hidden_states[-1]

                adapted = adapter(hidden_states)
                logits = head(adapted)
                log_probs = logits.log_softmax(dim=-1)

                input_lengths = torch.full(
                    (logits.size(0),), 
                    logits.size(1), 
                    dtype=torch.long,
                    device=device
                )
                target_lengths = attention_mask.sum(dim=1)

                loss = ctc_loss(
                    log_probs.transpose(0, 1),
                    labels,
                    input_lengths,
                    target_lengths
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Decode predictions
                pred_ids = log_probs.argmax(-1)
                for pred, label in zip(pred_ids, labels):
                    pred_str = processor.tokenizer.decode(pred, skip_special_tokens=True)
                    label_str = tokenizer.decode(label, skip_special_tokens=True)
                    hypotheses.append(pred_str)
                    references.append(label_str)

            # Synchronize and log metrics
            avg_loss = epoch_loss / len(train_loader)
            current_wer = wer(references, hypotheses)

            if rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs}")
                print(f"Train Loss: {avg_loss:.4f} | WER: {current_wer:.4f}")
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": avg_loss,
                    "train_wer": current_wer
                })

                # Validation
                val_loss, val_wer = validate(
                    adapter, 
                    head, 
                    whisper_model, 
                    val_loader, 
                    processor, 
                    tokenizer, 
                    ctc_loss, 
                    device
                )
                print(f"Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f}")
                wandb.log({
                    "val_loss": val_loss,
                    "val_wer": val_wer
                })

                # Save checkpoint
                if val_wer < best_wer:
                    best_wer = val_wer
                    torch.save({
                        'adapter': adapter.module.state_dict(),
                        'head': head.state_dict(),
                        'epoch': epoch,
                        'wer': val_wer,
                        'config': vars(args)
                    }, f"{args.adapter_type}_best.pt")

                # Unfreeze layers if WER threshold met
                if val_wer < args.unfreeze_wer_threshold:
                    print(f"WER threshold met ({val_wer:.4f} < {args.unfreeze_wer_threshold}), unfreezing last 4 Whisper layers")
                    for layer in whisper_model.model.encoder.layers[-4:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    
                    optimizer.add_param_group({
                        'params': [p for p in whisper_model.parameters() if p.requires_grad],
                        'lr': args.lr / 10
                    })
                        
                        # If using Qwen parameters:
                        # optimizer.add_param_group({
                        #     'params': [p for p in qwen_model.parameters() if p.requires_grad],
                        #     'lr': args.lr / 20
                        # })

     

            # except Exception as e:
            #     print(f"Rank {rank} encountered error in epoch {epoch}: {str(e)}")
            #     raise

    except Exception as e:
        print(f"Rank {rank} training failed: {str(e)}")
        raise
    finally:
        # Save model on rank 0
        if rank == 0 and dist.is_initialized():
            torch.save(adapter.module.state_dict(), f"{args.adapter_type}_adapter_final.pt")
            if 'wandb' in locals():
                wandb.save(f"{args.adapter_type}_adapter_final.pt")
                wandb.finish()
            print("Training complete.")
        
        # Ensure cleanup is called
        cleanup()

def validate(model, head, whisper_model, val_loader, processor, tokenizer, criterion, device):
    model.eval()
    whisper_model.eval()
    
    total_loss = 0.0
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in val_loader:
            input_features, labels, attention_mask = batch
            
            whisper_outputs = whisper_model(
                input_features,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = whisper_outputs.hidden_states[-1]
            
            adapted = model(hidden_states)
            logits = head(adapted)
            log_probs = logits.log_softmax(dim=-1)

            input_lengths = torch.full(
                (logits.size(0),),
                logits.size(1),
                dtype=torch.long,
                device=device
            )
            target_lengths = attention_mask.sum(dim=1)

            loss = criterion(
                log_probs.transpose(0, 1),
                labels,
                input_lengths,
                target_lengths
            )
            total_loss += loss.item()

            pred_ids = log_probs.argmax(-1)
            for pred, label in zip(pred_ids, labels):
                pred_str = processor.tokenizer.decode(pred, skip_special_tokens=True)
                label_str = tokenizer.decode(label, skip_special_tokens=True)
                hypotheses.append(pred_str)
                references.append(label_str)

    avg_loss = total_loss / len(val_loader)
    wer_score = wer(references, hypotheses)
    
    return avg_loss, wer_score


def main():
    # Initialize distributed environment
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_name", default="openai/whisper-large-v3")
    parser.add_argument("--qianwen_name", default="Qwen/Qwen-7B")
    parser.add_argument("--adapter_type", default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_threshold_wer", type=float, default=0.25)
    parser.add_argument("--unfreeze_after_wer", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=3)

        # Adapter-specific params
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--chunk_size", type=int, default=1600, help="Chunk size for streaming")
    args = parser.parse_args()

    # Start training
    train_adapter_ddp(rank, world_size, args)

if __name__ == "__main__":
    # Suppress the specific warning if desired
    warnings.filterwarnings("ignore", message=".*process group has NOT been destroyed before we destruct ProcessGroupNCCL.*")
    
    try:
        main()
    except Exception as e:
        print(f"Main process failed: {str(e)}")
        raise
    finally:
        # Final cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


