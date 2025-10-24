import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import dgl
import warnings
warnings.filterwarnings('ignore')

from models import FairnessAwareJobRecommender, get_model_config, create_user_job_graph
from utils import (
    DataPreprocessor, FairnessLoss, FairnessMetrics, 
    create_batch_data, plot_fairness_metrics, generate_synthetic_data
)


class JobRecommendationTrainer:
    """Main trainer class for the fairness-aware job recommendation model"""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = FairnessAwareJobRecommender(config).to(device)
        
        # Initialize loss function
        self.criterion = FairnessLoss(
            lambda_adv=config.get('lambda_adv', 1.0),
            lambda_fair=config.get('lambda_fair', 0.1)
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Initialize fairness metrics
        self.fairness_metrics = FairnessMetrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'demographic_parity': [],
            'equalized_odds': [],
            'exposure_gap': [],
            'adversary_accuracy': []
        }
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preprocessing data...")
        
        # Preprocess features
        df_processed = self.preprocessor.fit_transform(df)
        
        # Split features and targets
        feature_columns = [col for col in df_processed.columns if col != 'clicked']
        X = df_processed[feature_columns]
        y = df_processed['clicked']
        
        # Get sensitive attribute (gender)
        sensitive_attr = df_processed['user_gender']
        
        # Create train/validation split
        train_size = int(0.8 * len(df_processed))
        indices = np.random.permutation(len(df_processed))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
        sensitive_train, sensitive_val = sensitive_attr.iloc[train_indices], sensitive_attr.iloc[val_indices]
        
        # Create user-job graph
        print("Creating user-job interaction graph...")
        train_interactions = df_processed.iloc[train_indices]
        user_ids = train_interactions['user_id'].unique()
        job_ids = train_interactions['job_id'].unique()
        
        self.graph, self.user_to_idx, self.job_to_idx = create_user_job_graph(
            train_interactions, user_ids, job_ids
        )
        self.graph = self.graph.to(self.device)
        
        return X_train, X_val, y_train, y_val, sensitive_train, sensitive_val
    
    def create_dataloader(self, X, y, sensitive_attr, batch_size=256, shuffle=True):
        """Create DataLoader for training"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values)
        sensitive_tensor = torch.FloatTensor(sensitive_attr.values)
        
        dataset = TensorDataset(X_tensor, y_tensor, sensitive_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_click_loss = 0
        total_adv_loss = 0
        total_fair_loss = 0
        
        for batch_idx, (X_batch, y_batch, sensitive_batch) in enumerate(tqdm(train_loader, desc="Training")):
            self.optimizer.zero_grad()
            
            # Create batch data
            batch_df = pd.DataFrame(X_batch.numpy(), columns=self.preprocessor.feature_info.keys())
            batch_data = create_batch_data(batch_df, self.preprocessor, self.device)
            
            # Move targets to device
            y_batch = y_batch.to(self.device)
            sensitive_batch = sensitive_batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_data, self.graph)
            
            # Compute loss
            loss_dict = self.criterion(outputs, y_batch, sensitive_batch)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_click_loss += loss_dict['click_loss'].item()
            total_adv_loss += loss_dict['adversary_loss'].item()
            total_fair_loss += loss_dict['fairness_loss'].item()
        
        return {
            'total_loss': total_loss / len(train_loader),
            'click_loss': total_click_loss / len(train_loader),
            'adversary_loss': total_adv_loss / len(train_loader),
            'fairness_loss': total_fair_loss / len(train_loader)
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_sensitive = []
        
        with torch.no_grad():
            for X_batch, y_batch, sensitive_batch in tqdm(val_loader, desc="Validating"):
                # Create batch data
                batch_df = pd.DataFrame(X_batch.numpy(), columns=self.preprocessor.feature_info.keys())
                batch_data = create_batch_data(batch_df, self.preprocessor, self.device)
                
                # Move targets to device
                y_batch = y_batch.to(self.device)
                sensitive_batch = sensitive_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data, self.graph)
                
                # Compute loss
                loss_dict = self.criterion(outputs, y_batch, sensitive_batch)
                total_loss += loss_dict['total_loss'].item()
                
                # Store predictions for fairness metrics
                all_predictions.append(outputs['click_prob'].cpu())
                all_targets.append(y_batch.cpu())
                all_sensitive.append(sensitive_batch.cpu())
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_sensitive = torch.cat(all_sensitive, dim=0)
        
        # Compute fairness metrics
        demo_parity, _, _ = self.fairness_metrics.demographic_parity(all_predictions, all_sensitive)
        eq_odds, _, _, _, _ = self.fairness_metrics.equalized_odds(all_targets, all_predictions, all_sensitive)
        exposure_gap, _, _ = self.fairness_metrics.exposure_gap(all_predictions, all_sensitive)
        adv_accuracy = self.fairness_metrics.adversary_accuracy(all_predictions, all_sensitive)
        
        return {
            'val_loss': total_loss / len(val_loader),
            'demographic_parity': demo_parity,
            'equalized_odds': eq_odds,
            'exposure_gap': exposure_gap,
            'adversary_accuracy': adv_accuracy
        }
    
    def train(self, df, epochs=100, batch_size=256, patience=20):
        """Main training loop"""
        print("Preparing data...")
        X_train, X_val, y_train, y_val, sensitive_train, sensitive_val = self.prepare_data(df)
        
        # Create data loaders
        train_loader = self.create_dataloader(X_train, y_train, sensitive_train, batch_size, shuffle=True)
        val_loader = self.create_dataloader(X_val, y_val, sensitive_val, batch_size, shuffle=False)
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Store history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['demographic_parity'].append(val_metrics['demographic_parity'])
            self.history['equalized_odds'].append(val_metrics['equalized_odds'])
            self.history['exposure_gap'].append(val_metrics['exposure_gap'])
            self.history['adversary_accuracy'].append(val_metrics['adversary_accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Demographic Parity Gap: {val_metrics['demographic_parity']:.4f}")
            print(f"Equalized Odds Gap: {val_metrics['equalized_odds']:.4f}")
            print(f"Exposure Gap: {val_metrics['exposure_gap']:.4f}")
            print(f"Adversary Accuracy: {val_metrics['adversary_accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'preprocessor': self.preprocessor,
                    'config': self.config,
                    'history': self.history
                }, 'model.pt')
                print("Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
        
        print("\nTraining completed!")
        return self.history
    
    def load_model(self, model_path):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.preprocessor = checkpoint['preprocessor']
        self.history = checkpoint['history']
        print(f"Model loaded from {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Fairness-Aware Job Recommendation Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data', type=str, default='fairjob.csv', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lambda_adv', type=float, default=1.0, help='Adversarial loss weight')
    parser.add_argument('--lambda_fair', type=float, default=0.1, help='Fairness loss weight')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training...")
        
        # Load or generate data
        if os.path.exists(args.data):
            print(f"Loading data from {args.data}")
            df = pd.read_csv(args.data)
        else:
            print(f"Data file {args.data} not found. Generating synthetic data...")
            df = generate_synthetic_data(10000)
            df.to_csv(args.data, index=False)
            print(f"Synthetic data saved to {args.data}")
        
        # Update config with command line arguments
        config = get_model_config()
        config.update({
            'learning_rate': args.learning_rate,
            'lambda_adv': args.lambda_adv,
            'lambda_fair': args.lambda_fair
        })
        
        # Initialize trainer
        trainer = JobRecommendationTrainer(config)
        
        # Train model
        history = trainer.train(
            df, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            patience=args.patience
        )
        
        # Plot and save fairness metrics
        plot_fairness_metrics(history, 'fairness_metrics.png')
        print("Fairness metrics plot saved to fairness_metrics.png")
        
        # Save training history
        pd.DataFrame(history).to_csv('training_history.csv', index=False)
        print("Training history saved to training_history.csv")
        
        print("\nTraining completed successfully!")
        print("Model saved as 'model.pt'")
        print("You can now run the dashboard with: streamlit run app.py")
    
    else:
        print("Use --train flag to start training")
        print("Example: python train.py --train --epochs 50 --batch_size 256")


if __name__ == "__main__":
    main()
