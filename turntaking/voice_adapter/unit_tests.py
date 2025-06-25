import torch
from datasets import load_dataset
from transformers import WhisperProcessor, AutoTokenizer
from dataloader import collate_fn

# Initialize processor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B",trust_remote_code=True)

# --- Test 1: Verify Basic Tokenization ---
def test_basic_tokenization():
    test_texts = [
        'Hello world!',
        'This is a test with numbers 123.',
        '"Quoted text"',
        'Punctuation? Yes!',
        'Long text ' * 50  # Forces truncation
    ]
    
    print("=== Basic Tokenization Test ===")
    for text in test_texts:
        try:
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            decoded = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
            
            print(f"\nOriginal: {text}")
            print(f"Token IDs: {tokens.input_ids[0]}")
            print(f"Decoded: {decoded}")
            assert text.strip('"') in decoded, "Decoding mismatch!"
            
        except Exception as e:
            print(f"❌ Failed on: '{text}'\nError: {e}")
            raise

# --- Test 2: Verify Collate_fn Integration ---
def test_collate_fn():
    dummy_batch = [
        {"input_features": torch.randn(80, 3000), "text": "First sample text."},
        {"input_features": torch.randn(80, 3000), "text": 'Second "quoted" text!'},
    ]
    
    print("\n=== Collate Function Test ===")
    try:
        features, ids, masks = collate_fn(dummy_batch, processor, tokenizer)
        
        print("Shapes:")
        print(f"Features: {features.shape}")  # Should be [batch_size, 80, 3000]
        print(f"Token IDs: {ids.shape}")      # Should be [batch_size, seq_len]
        print(f"Masks: {masks.shape}")        # Should match token_ids
        
        # Verify decoded text
        decoded_texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        for original, decoded in zip([x["text"] for x in dummy_batch], decoded_texts):
            print(f"\nOriginal: {original}")
            print(f"Decoded: {decoded}")
            assert original.strip('"').rstrip('.') in decoded, "Text mismatch!"
            
    except Exception as e:
        print(f"❌ Collate failed: {e}")
        raise

# --- Test 3: Real Dataset Sample ---
def test_real_dataset_sample():
    dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train[:2]", trust_remote_code=True)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    print("\n=== Real Dataset Test ===")
    for sample in dataset:
        try:
            if sample["audio"] is None:
                continue
                
            audio = sample["audio"]["array"]
            text = sample["text"]
            
            # Test preprocessing
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            print(f"\nAudio shape: {inputs.input_features.shape}")
            print(f"Original text: {text}")
            print(f"Tokenized: {tokenized.input_ids[0]}")
            
            decoded = tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)
            assert text.strip('"').rstrip('.') in decoded, "Real sample decoding failed!"
            
        except Exception as e:
            print(f"❌ Failed on real sample: {e}")
            raise

def test_special_tokens():
    print("=== Special Token Test ===")
    
    # 1. Verify Special Token Existence
    required_special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",  # Start of sentence
        "eos_token": "[EOS]",  # End of sentence
    }
    
    for token_name, token_symbol in required_special_tokens.items():
        token = getattr(tokenizer, token_name, None)
        if token is None:
            print(f"❌ Missing {token_name}!")
        else:
            print(f"✅ {token_name}: {token} (ID: {tokenizer.convert_tokens_to_ids(token)})")
    
    # 2. Test Special Token Behavior
    test_cases = [
        ("", "Empty string → Should use [PAD] or [BOS]/[EOS]"),
        ("[UNK]", "Explicit unknown token"),
        ("Hello [UNK] world!", "Mixed unknown token"),
    ]
    
    for text, description in test_cases:
        print(f"\nTest Case: {description}")
        encoded = tokenizer(text, return_tensors="pt")
        decoded = tokenizer.decode(encoded.input_ids[0], skip_special_tokens=False)
        
        print(f"Original: {text}")
        print(f"Token IDs: {encoded.input_ids[0].tolist()}")
        print(f"Decoded (with special tokens): {decoded}")
        
        # Check if [UNK] appears when expected
        if "[UNK]" in text:
            assert tokenizer.unk_token_id in encoded.input_ids, "UNK token not inserted!"

# 3. Verify Padding in Batches
def test_padding():
    print("\n=== Padding Test ===")
    batch_texts = ["Short", "Very long text " * 20]
    encoded = tokenizer(batch_texts, padding=True, return_tensors="pt")
    
    print("Batch Token IDs:")
    for ids in encoded.input_ids:
        print(ids.tolist())
    
    # Check padding token is added to shorter sequence
    assert encoded.input_ids[0][-1] == tokenizer.pad_token_id, "Padding failed!"


if __name__ == "__main__":
    # test_special_tokens()
    test_padding()
    test_basic_tokenization()
    test_collate_fn()
    test_real_dataset_sample()
    print("\n✅ All tests passed!")