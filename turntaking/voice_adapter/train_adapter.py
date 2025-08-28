from html import parser
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
from voice_adapter import create_adapter_model

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
    

def train_adapter(args):
    # Initialize device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else 'cpu')
    
    # Initialize WandB
    if local_rank == 0:
        wandb.init(project="voice-adapter-training", config=vars(args))
    
    # Load unified model with adapter
    model = create_adapter_model(
        whisper_model_name=args.whisper_name,
        qianwen_model_name=args.qianwen_name,
        adapter_type=args.adapter_type,
        streaming_mode=args.streaming,
        target_seq_len=args.target_seq_len,  # Add this line

    ).to(device)
    tokenizer=model.tokenizer

    # Load processors
    processor = WhisperProcessor.from_pretrained(args.whisper_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.qianwen_name, trust_remote_code=True)
    # special_tokens_dict = {"additional_special_tokens": ["</audio>"]}
    # num_added = tokenizer.add_special_tokens(special_tokens_dict)
    # print(f"Added {num_added} special tokens to tokenizer: {special_tokens_dict['additional_special_tokens']}")
    # print(f"Tokenizer size after adding special tokens: {len(tokenizer)}")
    # # If you are using a model, resize its embeddings to match the new tokenizer size
    # model.resize_token_embeddings(len(tokenizer))
    # model=resize_token_embeddings(model,len(tokenizer))

    # Freeze base models
    for param in model.whisper.parameters():
        param.requires_grad = False
    for param in model.qianwen.parameters():
        param.requires_grad = False
    for param in model.adapter_module.parameters():
        param.requires_grad = True

    # Handle DDP setup if enabled
    if args.enable_ddp:
        # Initialize DDP
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        model = DDP(model, device_ids=[local_rank])
        print(f"DDP enabled for rank {local_rank}")
    else:
        print("Single GPU training mode")

    # Initialize datasets
    train_dataset = CommonVoiceDataset(split="train", max_samples=args.max_samples)
    val_dataset = CommonVoiceDataset(split="validation", max_samples=args.max_samples//10)

    # Create samplers based on DDP mode
    if args.enable_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b,  tokenizer,target_len=64),
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=(not args.enable_ddp)  # Only shuffle if not using DDP (sampler handles it)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b,  tokenizer,target_len=64),
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
    optimizer = torch.optim.AdamW(
        model.module.adapter_module.parameters() if args.enable_ddp else model.adapter_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    best_wer = float('inf')
    for epoch in range(args.epochs):
        if args.enable_ddp:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        references = []
        hypotheses = []

        for batch_idx, batch in enumerate(train_loader):
        
            input_features, labels, attention_mask = batch
            input_features = input_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
 
            # print("labels min:", labels.min().item(), "max:", labels.max().item())
            # print("labels shape:", labels.shape)
 
            # print("tokenizer vocab size", tokenizer.vocab_size)
            # print(len(tokenizer))

            # Reset cache if streaming
            if args.streaming and hasattr(model.module.adapter_module if args.enable_ddp else model.adapter_module, 'reset_cache'):
                (model.module if args.enable_ddp else model).adapter_module.reset_cache()
            # print("Trainable parameters:")
            # for name, param in model.module.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.shape)
            # Forward pass
            if args.enable_ddp:
                logits = model.module(input_features, attention_mask=attention_mask)
            else:
                logits = model(input_features,attention_mask=attention_mask)
          
  
            # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            # seq_len = labels.shape[1]
            assert labels.dtype == torch.long, f"Labels must be LongTensor, got {labels.dtype}"
            assert (labels < tokenizer.vocab_size).all() or (labels == -100).any(), "Some labels are out of range"
            # print("Logits vocab size:", logits.size(-1))
            # logits = logits[:, :seq_len, :]  # Truncate logits to match label length
            # Calculate loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            # print(loss.item())

            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

            epoch_loss += loss.item()

            # Decode predictions
            pred_ids = logits.argmax(-1)
            for pred, label in zip(pred_ids, labels):
                pred = [t for t in pred if t >= 0 and t != -100 and t < len(tokenizer)]
                pred_str = tokenizer.decode(pred, skip_special_tokens=True)
                label = [t for t in label if t >= 0 and t != -100 and t < len(tokenizer)]
                label_str = tokenizer.decode(label, skip_special_tokens=True)
                hypotheses.append(pred_str)
                references.append(label_str)

            if batch_idx % args.log_interval == 0 and local_rank == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss: {loss.item():.4f}")

        # Synchronize and log metrics
        avg_loss = epoch_loss / len(train_loader)
        current_wer = wer(references, hypotheses)

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {avg_loss:.4f} | WER: {current_wer:.4f}")
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_loss,
                "train_wer": current_wer
            })

            # Validation
            val_loss, val_wer = validate(
                model, 
                val_loader, 
                processor, 
                tokenizer, 
                criterion, 
                device,
                args.streaming,
                args.enable_ddp
            )
            print(f"Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "val_wer": val_wer
            })

            # Save checkpoint
            if val_wer < best_wer:
                best_wer = val_wer
                save_dict = {
                    'adapter': model.module.adapter_module.state_dict() if args.enable_ddp else model.adapter.state_dict(),
                    'epoch': epoch,
                    'wer': val_wer,
                    'config': vars(args)
        
                }
                torch.save(save_dict, f"{args.checkpoint_path}/{args.adapter_type}_best.pt")
                tokenizer.save_pretrained(f"{args.checkpoint_path}/{args.adapter_type}_tokenizer")

                # # Save checkpoint
                # if val_wer < best_wer:
                #     best_wer = val_wer
                #     torch.save({
                #         'adapter': model.module.state_dict(),
                #         'head': head.state_dict(),
                #         'epoch': epoch,
                #         'wer': val_wer,
                #         'config': vars(args)
                #     }, f"{args.adapter_type}_best.pt")


                if val_wer < args.freeze_threshold_wer:
                    print(f"WER threshold met, unfreezing last 4 Whisper layers")
                    whisper_model = model.module.whisper if args.enable_ddp else model.whisper
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

    if local_rank == 0:
        final_save_path = f"{args.checkpoint_path}/{args.adapter_type}_adapter_final.pt"
        torch.save(
            model.module.adapter.state_dict() if args.enable_ddp else model.adapter.state_dict(),
            final_save_path
        )
        if 'wandb' in locals():
            wandb.save(final_save_path)
            wandb.finish()
        print("Training complete.")

    if args.enable_ddp:
        # Ensure cleanup is called
        cleanup()

def validate(model, val_loader, processor, tokenizer, criterion, device, streaming, enable_ddp):
    model.eval()
    val_loss = 0.0
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_features, labels, attention_mask = batch
            input_features = input_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            
            if streaming and hasattr(model.module.adapter_module if enable_ddp else model.adapter_module, 'reset_cache'):
                (model.module if enable_ddp else model).adapter_module.reset_cache()
                
            if enable_ddp:
                logits = model.module(input_features, attention_mask=attention_mask)
            else:
                logits = model(input_features,attention_mask=attention_mask)
            # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            # seq_len = labels.shape[1]
            # logits = logits[:, :seq_len, :]  # Truncate logits to match label length
            # Calculate loss
            # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            val_loss += loss.item()
            
            pred_ids = logits.argmax(-1)
            for pred, label in zip(pred_ids, labels):
                pred = [t for t in pred if t >= 0 and t != -100 and t < len(tokenizer)]
                pred_str = tokenizer.decode(pred, skip_special_tokens=True)
                label = [t for t in label if t >= 0 and t != -100 and t < len(tokenizer)]
                label_str = tokenizer.decode(label, skip_special_tokens=True)
                hypotheses.append(pred_str)
                references.append(label_str)
    
    avg_loss = val_loss / len(val_loader)
    wer_score = wer(references, hypotheses)
    
    return avg_loss, wer_score


def main():

    
    # Initialize distributed environment
    rank = int(os.environ.get('RANK', 0))
    # local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_name", default="openai/whisper-large-v3")
    parser.add_argument("--qianwen_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter_type", default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_threshold_wer", type=float, default=0.25)
    parser.add_argument("--unfreeze_after_wer", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument('--enable_ddp', action='store_true', help='Enable DistributedDataParallel')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
        # Adapter-specific params
    parser.add_argument('--target_seq_len', type=int, default=None, help='Target sequence length for sequence adapters')
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--chunk_size", type=int, default=1600, help="Chunk size for streaming")
    parser.add_argument('--log-interval', type=int, default=10, help='How often to log training status')
    parser.add_argument('--checkpoint_path', type=str, default='/home/rmfrieske/turntaking/turntaking/checkpoints', help='Path to save/load the adapter checkpoint')
    args = parser.parse_args()

    # Start training
    train_adapter( args)

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


