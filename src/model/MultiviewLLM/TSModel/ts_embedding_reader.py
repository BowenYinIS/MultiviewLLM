import numpy as np
import os

def read_embeddings(embeddings_path):
    """Read the embeddings file"""
    print(f"Loading embeddings from {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        print(f"Error: File {embeddings_path} does not exist!")
        return None
    
    embeddings = np.load(embeddings_path, allow_pickle=True)
    
    print(f"Type: {type(embeddings)}")
    print(f"Shape: {embeddings.shape}")
    print(f"Number of samples: {len(embeddings)}")
    
    # Look at the first few samples
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(embeddings))):
        sample = embeddings[i]
        print(f"\nSample {i+1}:")
        print(f"  act_idn_sky: {sample['act_idn_sky']}")
        print(f"  embeddings shape: {sample['embeddings'].shape}")
        print(f"  sequence_length: {sample['sequence_length']}")
        print(f"  target_delinquency: {sample['target_delinquency']}")
        print(f"  First 5 embedding values: {sample['embeddings'][0, :5]}")
    
    return embeddings

if __name__ == "__main__":
    # Path to embeddings file
    embeddings_path = "/data/mjmao/credit/MultiviewLLM/checkpoint/MultiviewLLM/TSModel/ts_embeddings.npy"
    
    # Read embeddings
    embeddings = read_embeddings(embeddings_path)
    # e.g., embeddings[0]['embeddings'].shape = sequence_length * hidden_dim = 59 * 256
