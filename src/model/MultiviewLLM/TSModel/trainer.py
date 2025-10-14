import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import TimeSeriesTransformer, create_model, apply_masking
from dataloader import create_dataloader, MultiviewTimeSeriesDataset

class SelfSupervisedTrainer:
    """Trainer for self-supervised learning on time series data"""
    
    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-5):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.mcc_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Classification for MCC classes
        self.amt_criterion = nn.MSELoss()  # Regression for amounts
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_mcc_losses = []
        self.train_amt_losses = []
        self.val_mcc_losses = []
        self.val_amt_losses = []
        
    def compute_loss(self, predictions, targets, mask_positions, attention_mask):
        """
        Compute self-supervised loss for masked tokens
        Returns separate losses for MCC and amount predictions
        """
        from model import mcc_to_class
        
        batch_size = predictions['mcc_predictions'].shape[0]
        total_mcc_loss = 0.0
        total_amt_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            seq_len = (~attention_mask[i]).sum().item()
            if seq_len == 0:
                continue
                
            # Get masked positions for this sample
            if isinstance(mask_positions, list) and len(mask_positions) > i:
                sample_mask_positions = mask_positions[i]
            else:
                sample_mask_positions = mask_positions
                
            if len(sample_mask_positions) == 0:
                continue
                
            # Only compute loss on masked positions that are valid (not padding)
            valid_mask_positions = [pos for pos in sample_mask_positions if pos < seq_len]
            
            if len(valid_mask_positions) == 0:
                continue
                
            # Prepare MCC targets (convert to classes)
            mcc_targets = []
            for pos in valid_mask_positions:
                mcc_code = targets['mcc_cde'][i][pos].item()
                mcc_class = mcc_to_class(mcc_code)
                mcc_targets.append(mcc_class)
            
            # Filter out masked positions (-1) for MCC loss
            valid_mcc_positions = []
            valid_mcc_targets = []
            for j, pos in enumerate(valid_mask_positions):
                if mcc_targets[j] != -1:  # Not a masked token
                    valid_mcc_positions.append(pos)
                    valid_mcc_targets.append(mcc_targets[j])
            
            # Compute MCC loss (classification)
            if len(valid_mcc_positions) > 0:
                mcc_pred = predictions['mcc_predictions'][i][valid_mcc_positions]
                mcc_target_tensor = torch.tensor(valid_mcc_targets, dtype=torch.long, device=mcc_pred.device)
                mcc_loss = self.mcc_criterion(mcc_pred, mcc_target_tensor)
                total_mcc_loss += mcc_loss
            
            # Compute amount loss (regression) - amounts are already log-transformed and normalized in dataloader
            amt_pred = predictions['amt_predictions'][i][valid_mask_positions]
            amt_target = targets['txn_amt'][i][valid_mask_positions]
            amt_loss = self.amt_criterion(amt_pred, amt_target)
            
            total_amt_loss += amt_loss
            
            valid_samples += 1
            
        # Average losses across all valid samples
        avg_mcc_loss = total_mcc_loss / max(valid_samples, 1)
        avg_amt_loss = total_amt_loss / max(valid_samples, 1)
        total_loss = avg_mcc_loss + avg_amt_loss
        
        return {
            'total_loss': total_loss,
            'mcc_loss': avg_mcc_loss,
            'amt_loss': avg_amt_loss
        }
    
    def train_epoch(self, dataloader, mask_prob=0.15):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_mcc_loss = 0.0
        total_amt_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Move batch to device
            batch_on_device = self._move_batch_to_device(batch)
            
            # Apply masking
            masked_batch, mask_positions = apply_masking(
                batch_on_device, 
                mask_prob=mask_prob, 
                device=self.device
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model.forward_with_predictions(masked_batch, mask_prob)
            
            # Compute loss
            loss_dict = self.compute_loss(
                predictions, 
                batch_on_device, 
                mask_positions, 
                predictions['attention_mask']
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss'].item()
            total_mcc_loss += loss_dict['mcc_loss'].item()
            total_amt_loss += loss_dict['amt_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'total': f'{loss_dict["total_loss"].item():.4f}',
                'mcc': f'{loss_dict["mcc_loss"].item():.4f}',
                'amt': f'{loss_dict["amt_loss"].item():.4f}'
            })
            
        return {
            'total_loss': total_loss / num_batches,
            'mcc_loss': total_mcc_loss / num_batches,
            'amt_loss': total_amt_loss / num_batches
        }
    
    def validate(self, dataloader, mask_prob=0.15):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_mcc_loss = 0.0
        total_amt_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch_on_device = self._move_batch_to_device(batch)
                
                # Apply masking
                masked_batch, mask_positions = apply_masking(
                    batch_on_device, 
                    mask_prob=mask_prob, 
                    device=self.device
                )
                
                # Forward pass
                predictions = self.model.forward_with_predictions(masked_batch, mask_prob)
                
                # Compute loss
                loss_dict = self.compute_loss(
                    predictions, 
                    batch_on_device, 
                    mask_positions, 
                    predictions['attention_mask']
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_mcc_loss += loss_dict['mcc_loss'].item()
                total_amt_loss += loss_dict['amt_loss'].item()
                num_batches += 1
                
        return {
            'total_loss': total_loss / num_batches,
            'mcc_loss': total_mcc_loss / num_batches,
            'amt_loss': total_amt_loss / num_batches
        }
    
    def _move_batch_to_device(self, batch):
        """Move batch data to device"""
        batch_on_device = {}
        for key, value in batch.items():
            if key in ['mcc_cde', 'hod', 'dow', 'wom', 'moy', 'txn_amt', 'target']:
                if key == 'target':
                    batch_on_device[key] = value.to(self.device)
                else:
                    # For lists of tensors, move each tensor to device
                    batch_on_device[key] = [tensor.to(self.device) for tensor in value]
            else:
                batch_on_device[key] = value  # Keep non-tensor data as is
        return batch_on_device
    
    def save_checkpoint(self, epoch, loss, save_dir, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        if is_best:
            # Save as best checkpoint
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint_best.pt'))
        else:
            # Save as regular checkpoint
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    
    def plot_training_curves(self, save_dir):
        """Plot training and validation loss curves"""
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Total Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Total Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Val Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: MCC Loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_mcc_losses, 'b-', label='Train MCC Loss')
        plt.plot(epochs, self.val_mcc_losses, 'r-', label='Val MCC Loss')
        plt.title('MCC Classification Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Amount Loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_amt_losses, 'b-', label='Train Amount Loss')
        plt.plot(epochs, self.val_amt_losses, 'r-', label='Val Amount Loss')
        plt.title('Amount Regression Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: All losses together
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.train_mcc_losses, 'b-', label='Train MCC', alpha=0.7)
        plt.plot(epochs, self.train_amt_losses, 'g-', label='Train Amount', alpha=0.7)
        plt.plot(epochs, self.val_mcc_losses, 'r--', label='Val MCC', alpha=0.7)
        plt.plot(epochs, self.val_amt_losses, 'orange', linestyle='--', label='Val Amount', alpha=0.7)
        plt.title('All Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")

def main():
    parser = argparse.ArgumentParser(description='Self-Supervised Training for Time Series Transformer')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, 
                       default="data/processed_data/ts_processed_data/multivariate_timeseries_train.jsonl",
                       help='Path to training data')
    parser.add_argument('--val_file', type=str,
                       default="data/processed_data/ts_processed_data/multivariate_timeseries_test.jsonl",
                       help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--mask_prob', type=float, default=0.15,
                       help='Masking probability')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda, cpu)')
    
    # Save arguments
    parser.add_argument('--save_dir', type=str, default='checkpoint/MultiviewLLM/TSModel',
                       help='Directory to save checkpoints')
    parser.add_argument('--plot_dir', type=str, default='Fig/TSTraining',
                       help='Directory to save training plots')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create training dataset to compute normalization statistics
    print("Loading training data and computing normalization statistics...")
    train_dataset = MultiviewTimeSeriesDataset(args.train_file)
    amt_mean, amt_std = train_dataset.compute_amount_stats()
    
    # Create data loaders with normalization
    train_dataloader = create_dataloader(
        file_path=args.train_file,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        amt_mean=amt_mean,
        amt_std=amt_std
    )
    
    val_dataloader = create_dataloader(
        file_path=args.val_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        amt_mean=amt_mean,
        amt_std=amt_std
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        device=device
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = SelfSupervisedTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_losses = trainer.train_epoch(train_dataloader, mask_prob=args.mask_prob)
        print(f"Train - Total: {train_losses['total_loss']:.4f}, MCC: {train_losses['mcc_loss']:.4f}, Amount: {train_losses['amt_loss']:.4f}")
        
        # Validate
        val_losses = trainer.validate(val_dataloader, mask_prob=args.mask_prob)
        print(f"Val   - Total: {val_losses['total_loss']:.4f}, MCC: {val_losses['mcc_loss']:.4f}, Amount: {val_losses['amt_loss']:.4f}")
        
        # Track losses for plotting
        trainer.train_losses.append(train_losses['total_loss'])
        trainer.val_losses.append(val_losses['total_loss'])
        trainer.train_mcc_losses.append(train_losses['mcc_loss'])
        trainer.train_amt_losses.append(train_losses['amt_loss'])
        trainer.val_mcc_losses.append(val_losses['mcc_loss'])
        trainer.val_amt_losses.append(val_losses['amt_loss'])
        
        # Save only the best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            trainer.save_checkpoint(epoch + 1, val_losses['total_loss'], args.save_dir, is_best=True)
            print(f"New best model saved as checkpoint_best.pt with val_loss: {val_losses['total_loss']:.4f}")
    
    # Plot training curves
    print("\nGenerating training curves...")
    trainer.plot_training_curves(args.plot_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()