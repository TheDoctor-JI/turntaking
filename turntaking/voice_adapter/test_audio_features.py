import pytest
import torch
import numpy as np
from unittest.mock import patch
from datasets import Dataset
from transformers import WhisperProcessor
from dataloader import CommonVoiceDataset, collate_fn
from voice_adapter import UnifiedModelWithAdapter
from transformers import WhisperProcessor, AutoTokenizer
from adapter_modules import CNNAdapter, LinearAdapter, LinearAttentionAdapter
# Mock dataset with valid/invalid audio samples
def create_mock_dataset():
    return Dataset.from_dict({
        "audio": [
            # Valid audio (48kHz sine wave)
            {"array": np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), "sampling_rate": 16000},
            # Invalid audio (wrong sample rate)
            {"array": np.random.randn(8000), "sampling_rate": 8000},
            # Empty audio
            {"array": np.array([]), "sampling_rate": 16000},
        ],
        "text": ["Test 1", "Test 2", "Test 3"]
    })

@pytest.fixture
def processor():
    return WhisperProcessor.from_pretrained("openai/whisper-large-v3")

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# @pytest.fixture
# def model():
#     return UnifiedModelWithAdapter.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

# def test_real_sample_processing(processor, tokenizer):
#     """Test processing of real samples from Common Voice"""
#     # Create dataset with just 2 samples for testing
#     dataset = CommonVoiceDataset(
#         split="train",  # First 2 samples
#         processor=processor,
#         max_samples=2
#     )

#     samples = list(dataset)[:2]  # Convert iterable to list

#     # Verify we got exactly 2 samples
#     assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"

#     for sample in samples:
#         # Check feature dimensions
#         assert sample["input_features"].shape == (128, 3000), \
#             f"Unexpected feature shape: {sample['input_features'].shape}"

#         # Check text processing
#         assert isinstance(sample["text"], str), "Text should be a string"
#         assert len(sample["text"]) > 0, "Text should not be empty"

#         # Verify no NaN values
#         assert not torch.isnan(sample["input_features"]).any(), "Features contain NaN values"

#         # Sample rate check (assuming processor has a way to verify sample rate)
#         # This is a conceptual check; actual implementation depends on processor details
#         # For example, you might have a way to check the sample rate of the processed audio
#         # Here we assume processor or inputs can give us this information

#         # Check if the processor processed the audio at the correct sample rate
#         # Assuming `processor` has a method or attribute to check sample rate
#         assert processor.feature_extractor.sampling_rate == 16000, "Audio is not processed at 16000 Hz" "Audio is not processed at 16000 Hz"

# def test_collate_fn_with_real_samples(processor, tokenizer):
#     """Test batch processing with real samples"""
#     dataset = CommonVoiceDataset(
#         split="train",  # First 2 samples
#         processor=processor,
#         max_samples=2
#     )
    
#     # Create a batch of samples
#     batch = list(dataset)[:2] 
    
#     # Process through collate_fn
#     features, ids, masks = collate_fn(
#         batch,
#         processor=processor,
#         tokenizer=tokenizer,
#         device="cpu"
#     )
    
#     # Verify batch dimensions
#     assert features.shape == (2, 128, 3000), "Feature batch dimension mismatch"
#     assert ids.shape[0] == 2, "Text batch dimension mismatch"
#     assert masks.shape[0] == 2, "Mask batch dimension mismatch"
    
#     # Verify padding
#     assert (masks == 1).any(), "All masks are zero (no valid tokens)"
#     assert torch.all(features[0] != 0), "Features appear to be zero-padded incorrectly"

# def test_streaming_behavior(processor):
#     """Verify streaming mode works correctly"""
#     dataset = CommonVoiceDataset(
#         split="train",
#         processor=processor,
#         max_samples=5,
#         num_shards=1
#     )
    
#     count = 0
#     for sample in dataset:
#         count += 1
#         assert sample["input_features"].shape == (128, 3000), "Feature shape mismatch"
#         if count >= 5:
#             break
    
#     assert count == 5, f"Expected 5 samples, got {count}"

# def test_text_preprocessing(processor):
#     """Verify text preprocessing rules"""
#     test_cases = [
#         ('"Hello"', "Hello."),
#         ('No punctuation', "No punctuation."),
#         ('Existing!', "Existing!"),
#         ('"Quoted."', "Quoted.")
#     ]
    
#     dataset = CommonVoiceDataset(processor=processor)
    
#     for input_text, expected in test_cases:
#         # Mock an item with our test text
#         mock_item = {
#             "audio": {"array": np.zeros(16000), "sampling_rate": 16000},
#             "sentence": input_text
#         }
        
#         # Process through the dataset's text preprocessing
#         processed = dataset._preprocess_text(mock_item["sentence"])
#         assert processed == expected, \
#             f"Preprocessing failed: '{input_text}' -> '{processed}' (expected '{expected}')"

# def test_invalid_samples(processor):
#     """Verify handling of invalid samples"""
#     invalid_item = {
#         "audio": None,
#         "sentence": "This should be skipped"
#     }
#     # Patch the data source used by CommonVoiceDataset
#     with patch('turntaking.voice_adapter.dataloader.load_dataset', return_value=[invalid_item]):
#         dataset = CommonVoiceDataset(processor=processor)
#         has_invalid = False
#         for sample in dataset:
#             if sample["text"] == "This should be skipped":
#                 has_invalid = True
#                 break
#         assert not has_invalid, "Invalid sample was not skipped"

# def test_labels_tokenization_and_vocab(processor,tokenizer):
#     """Test that labels are tokenized correctly and within vocabulary size."""
#     dataset = CommonVoiceDataset(
#         split="train",
#         processor=processor,
#         max_samples=10
#     )
#     batch = [sample for sample in dataset]
#     # Use your collate_fn to get input_features, input_ids, attention_mask
#     input_features, input_ids, attention_mask = collate_fn(
#         batch, processor, tokenizer, device="cpu"
#     )
#     labels = input_ids  # or however your pipeline defines labels

#     print("labels min:", labels.min().item(), "max:", labels.max().item(), "dtype:", labels.dtype)
#     print("vocab size:", tokenizer.vocab_size)
#     # print("vocab size:", model.config.vocab_size)

#     # Check dtype
#     assert labels.dtype in (torch.int32, torch.int64), "Labels must be integer dtype"
#     # Check value range (allow -100 for ignore_index)
#     assert labels.min() >= -100, "Labels contain values less than -100"
#     assert labels.max() < tokenizer.vocab_size, "Labels contain values >= vocab size"
#     # Optionally, check normalization (no float labels)
#     assert torch.all(labels == labels.long()), "Labels are not properly normalized to integers"


def test_tokenizer_and_special_token_ids(tokenizer):
    """
    Test that all special tokens used by the tokenizer have IDs < len(tokenizer),
    and that the model's embedding matrix matches the tokenizer size.
    """
 
    vocab_size = len(tokenizer)
    special_tokens = tokenizer.all_special_tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    print("Tokenizer vocab size:", vocab_size)
    print("Special tokens:", special_tokens)
    print("Special token IDs:", special_token_ids)

    # Check that all special token IDs are within the embedding matrix
    for token, token_id in zip(special_tokens, special_token_ids):
        assert 0 <= token_id < vocab_size, (
            f"Special token '{token}' has out-of-range ID {token_id} (vocab_size={vocab_size})"
        )

    # Check that model's embedding matches tokenizer size
    # model_emb_size = model.get_input_embeddings().weight.shape[0]
    # assert model_emb_size == vocab_size, (
    #     f"Model embedding size ({model_emb_size}) does not match tokenizer size ({vocab_size})"
    # )

    # Optionally, check that special tokens can be generated by the tokenizer
    for token in special_tokens:
        generated_id = tokenizer.convert_tokens_to_ids(token)
        assert generated_id != tokenizer.unk_token_id, (
            f"Special token '{token}' is not recognized by the tokenizer."
        )
if __name__ == "__main__":
    pytest.main([__file__, "-v"])