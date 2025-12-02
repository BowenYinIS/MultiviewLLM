#!/usr/bin/env python3
"""
Test script to read and display the npy file.
"""

import numpy as np
import sys
import os

def test_read_npy():
    """Test reading the npy file and show information."""
    npy_path = 'checkpoint/MultiviewLLM/TSModel/samples_min6mo_fixed_2test.npy'
    
    print(f"Reading npy file: {npy_path}")
    
    # Load the npy file
    batch_embeddings = np.load(npy_path, allow_pickle=True).item()
    
    print(f"Type: {type(batch_embeddings)}")
    print(f"Number of samples: {len(batch_embeddings)}")
    print(f"Number of keys: {len(batch_embeddings.keys())}")
    
    # Show key statistics
    keys = list(batch_embeddings.keys())
    print(f"Key range: {min(keys)} to {max(keys)}")
    print(f"All keys are consecutive: {keys == list(range(len(keys)))}")
    
    # Show first 30 embeddings and their dimensions
    keys_subset = keys[:30]
    
    for i, key in enumerate(keys_subset):
        embedding = batch_embeddings[key]
        print(f"Embedding {i+1}: shape = {embedding.shape}")
    
if __name__ == "__main__":
    test_read_npy()
