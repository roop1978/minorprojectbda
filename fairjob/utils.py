import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class FairnessMetrics:
    """Class for computing various fairness metrics"""
    
    @staticmethod
    def demographic_parity(y_pred, sensitive_attr):
        """
        Compute demographic parity gap
        P(Y=1|A=0) - P(Y=1|A=1)
        """
        y_pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        sensitive_np = sensitive_attr.detach().cpu().numpy() if torch.is_tensor(sensitive_attr) else sensitive_attr
        
        # Convert to binary predictions if probabilities
        if y_pred_np.max() <= 1.0 and y_pred_np.min() >= 0.0:
            y_binary = (y_pred_np > 0.5).astype(int)
        else:
            y_binary = y_pred_np
            
        prob_group_0 = np.mean(y_binary[sensitive_np == 0])
        prob_group_1 = np.mean(y_binary[sensitive_np == 1])
        
        return prob_group_0 - prob_group_1, prob_group_0, prob_group_1
    
    @staticmethod
    def equalized_odds(y_true, y_pred, sensitive_attr):
        """
        Compute equalized odds gap
        |P(Y=1|Y_true=1, A=0) - P(Y=1|Y_true=1, A=1)| + 
        |P(Y=1|Y_true=0, A=0) - P(Y=1|Y_true=0, A=1)|
        """
        y_true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        sensitive_np = sensitive_attr.detach().cpu().numpy() if torch.is_tensor(sensitive_attr) else sensitive_attr
        
        if y_pred_np.max() <= 1.0 and y_pred_np.min() >= 0.0:
            y_binary = (y_pred_np > 0.5).astype(int)
        else:
            y_binary = y_pred_np
            
        # True positive rates
        tpr_0 = np.mean(y_binary[(y_true_np == 1) & (sensitive_np == 0)])
        tpr_1 = np.mean(y_binary[(y_true_np == 1) & (sensitive_np == 1)])
        
        # False positive rates  
        fpr_0 = np.mean(y_binary[(y_true_np == 0) & (sensitive_np == 0)])
        fpr_1 = np.mean(y_binary[(y_true_np == 0) & (sensitive_np == 1)])
        
        tpr_gap = abs(tpr_0 - tpr_1)
        fpr_gap = abs(fpr_0 - fpr_1)
        
        return tpr_gap + fpr_gap, tpr_0, tpr_1, fpr_0, fpr_1
    
    @staticmethod
    def exposure_gap(y_pred, sensitive_attr, top_k=10):
        """
        Compute exposure gap - difference in average predicted scores
        between demographic groups in top-k recommendations
        """
        y_pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        sensitive_np = sensitive_attr.detach().cpu().numpy() if torch.is_tensor(sensitive_attr) else sensitive_attr
        
        # Get top-k predictions
        top_k_indices = np.argsort(y_pred_np)[-top_k:]
        
        exposure_group_0 = np.mean(y_pred_np[(top_k_indices) & (sensitive_np[top_k_indices] == 0)])
        exposure_group_1 = np.mean(y_pred_np[(top_k_indices) & (sensitive_np[top_k_indices] == 1)])
        
        return exposure_group_0 - exposure_group_1, exposure_group_0, exposure_group_1
    
    @staticmethod
    def adversary_accuracy(adversary_pred, sensitive_attr):
        """Compute adversary's accuracy in predicting sensitive attribute"""
        adversary_np = adversary_pred.detach().cpu().numpy() if torch.is_tensor(adversary_pred) else adversary_pred
        sensitive_np = sensitive_attr.detach().cpu().numpy() if torch.is_tensor(sensitive_attr) else sensitive_attr
        
        if adversary_np.ndim > 1:
            adversary_pred_binary = np.argmax(adversary_np, axis=1)
        else:
            adversary_pred_binary = (adversary_np > 0.5).astype(int)
            
        return accuracy_score(sensitive_np, adversary_pred_binary)


class DataPreprocessor:
    """Class for preprocessing job recommendation data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_info = {}
        
    def fit_transform(self, df):
        """Fit preprocessors and transform data"""
        df_processed = df.copy()
        
        # Separate features by type
        categorical_features = ['user_id', 'job_id', 'job_category', 'user_education', 
                               'user_experience_level', 'job_location', 'job_type', 'job_company_size']
        numerical_features = ['user_age', 'job_salary', 'job_required_experience', 
                            'user_skill_match', 'session_duration', 'session_position',
                            'user_click_prob', 'job_popularity', 'user_job_history',
                            'time_of_day', 'day_of_week', 'user_activity_level', 'job_freshness']
        
        # Process categorical features
        for feature in categorical_features:
            if feature in df_processed.columns:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
                self.feature_info[feature] = {
                    'type': 'categorical',
                    'unique_values': len(self.label_encoders[feature].classes_)
                }
        
        # Process numerical features
        for feature in numerical_features:
            if feature in df_processed.columns:
                # Skip if already processed as categorical
                if feature not in self.label_encoders:
                    if feature not in self.scalers:
                        self.scalers[feature] = StandardScaler()
                    df_processed[feature] = self.scalers[feature].fit_transform(
                        df_processed[feature].values.reshape(-1, 1)
                    ).flatten()
                    self.feature_info[feature] = {
                        'type': 'numerical',
                        'mean': self.scalers[feature].mean_[0],
                        'std': self.scalers[feature].scale_[0]
                    }
        
        return df_processed
    
    def transform(self, df):
        """Transform new data using fitted preprocessors"""
        df_processed = df.copy()
        
        # Transform categorical features
        for feature, encoder in self.label_encoders.items():
            if feature in df_processed.columns:
                # Handle unseen categories
                df_processed[feature] = df_processed[feature].astype(str)
                mask = df_processed[feature].isin(encoder.classes_)
                df_processed.loc[~mask, feature] = encoder.classes_[0]  # Use first class for unseen
                df_processed[feature] = encoder.transform(df_processed[feature])
        
        # Transform numerical features
        for feature, scaler in self.scalers.items():
            if feature in df_processed.columns:
                # Skip if already processed as categorical
                if feature not in self.label_encoders:
                    df_processed[feature] = scaler.transform(
                        df_processed[feature].values.reshape(-1, 1)
                    ).flatten()
        
        return df_processed


class FairnessLoss:
    """Custom loss function combining click prediction, adversarial, and fairness losses"""
    
    def __init__(self, lambda_adv=1.0, lambda_fair=0.1):
        self.lambda_adv = lambda_adv
        self.lambda_fair = lambda_fair
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def __call__(self, outputs, targets, sensitive_attr):
        """
        Compute total loss: L_total = L_click - λ_adv * L_adv + λ_fair * L_fair
        """
        click_prob = outputs['click_prob'].squeeze()
        adversary_out = outputs['adversary_out']
        fairness_out = outputs['fairness_out']
        
        # Click prediction loss
        click_loss = self.bce_loss(click_prob, targets.float())
        
        # Adversarial loss (we want adversary to fail)
        adversary_loss = self.ce_loss(adversary_out, sensitive_attr.long())
        
        # Fairness regularization loss (penalize demographic differences)
        fairness_loss = self.mse_loss(fairness_out.squeeze(), torch.zeros_like(fairness_out.squeeze()))
        
        # Total loss with adversarial debiasing
        total_loss = click_loss - self.lambda_adv * adversary_loss + self.lambda_fair * fairness_loss
        
        return {
            'total_loss': total_loss,
            'click_loss': click_loss,
            'adversary_loss': adversary_loss,
            'fairness_loss': fairness_loss
        }


def create_batch_data(df_batch, preprocessor, device):
    """Create batch data for model input"""
    
    # Extract features
    wide_features = df_batch[['user_age', 'job_salary', 'user_skill_match', 
                             'job_required_experience', 'job_company_size',
                             'session_duration', 'user_education', 'job_category',
                             'user_experience_level', 'job_type']].values
    
    deep_features = df_batch[['user_age', 'job_salary', 'user_skill_match', 
                             'job_required_experience', 'job_company_size',
                             'session_duration', 'user_education', 'job_category',
                             'user_experience_level', 'job_type', 'job_location',
                             'user_gender', 'user_click_prob', 'job_popularity',
                             'user_job_history', 'session_position', 'time_of_day',
                             'day_of_week', 'user_activity_level', 'job_freshness']].values
    
    user_features = df_batch[['user_age', 'user_education', 'user_experience_level',
                             'user_skill_match', 'user_click_prob', 'user_job_history',
                             'user_activity_level', 'user_gender', 'session_duration',
                             'session_position', 'time_of_day', 'day_of_week',
                             'job_popularity', 'user_job_history', 'user_activity_level']].values
    
    job_features = df_batch[['job_salary', 'job_required_experience', 'job_company_size',
                            'job_category', 'job_type', 'job_location', 'job_popularity',
                            'job_freshness', 'job_category', 'job_type', 'job_location',
                            'job_popularity']].values
    
    # Convert to tensors
    batch_data = {
        'wide_features': torch.FloatTensor(wide_features).to(device),
        'deep_features': torch.FloatTensor(deep_features).to(device),
        'user_features': torch.FloatTensor(user_features).to(device),
        'job_features': torch.FloatTensor(job_features).to(device)
    }
    
    return batch_data


def plot_fairness_metrics(metrics_history, save_path=None):
    """Plot fairness metrics over training epochs"""
    
    epochs = range(1, len(metrics_history['demographic_parity']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Demographic Parity
    axes[0, 0].plot(epochs, metrics_history['demographic_parity'])
    axes[0, 0].set_title('Demographic Parity Gap')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Parity Gap')
    axes[0, 0].grid(True)
    
    # Equalized Odds
    axes[0, 1].plot(epochs, metrics_history['equalized_odds'])
    axes[0, 1].set_title('Equalized Odds Gap')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Odds Gap')
    axes[0, 1].grid(True)
    
    # Exposure Gap
    axes[1, 0].plot(epochs, metrics_history['exposure_gap'])
    axes[1, 0].set_title('Exposure Gap')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Exposure Gap')
    axes[1, 0].grid(True)
    
    # Adversary Accuracy
    axes[1, 1].plot(epochs, metrics_history['adversary_accuracy'])
    axes[1, 1].set_title('Adversary Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_fairness_dashboard(metrics_history):
    """Create interactive Plotly dashboard for fairness metrics"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Demographic Parity Gap', 'Equalized Odds Gap', 
                       'Exposure Gap', 'Adversary Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = list(range(1, len(metrics_history['demographic_parity']) + 1))
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_history['demographic_parity'], 
                  name='Demographic Parity', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_history['equalized_odds'], 
                  name='Equalized Odds', line=dict(color='red')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_history['exposure_gap'], 
                  name='Exposure Gap', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=metrics_history['adversary_accuracy'], 
                  name='Adversary Accuracy', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, 
                     title_text="Fairness Metrics Over Training")
    
    return fig


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic FairJob dataset for testing"""
    
    np.random.seed(42)
    
    # User features
    user_ids = np.random.randint(1, 1001, n_samples)
    user_ages = np.random.normal(35, 10, n_samples).astype(int)
    user_ages = np.clip(user_ages, 22, 65)
    
    # Gender (0: male, 1: female) - introduce bias
    user_genders = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Education levels
    user_educations = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_samples, p=[0.2, 0.5, 0.25, 0.05])
    
    # Experience levels
    user_experience_levels = np.random.choice(['Entry', 'Mid', 'Senior', 'Executive'], 
                                            n_samples, p=[0.3, 0.4, 0.25, 0.05])
    
    # Job features
    job_ids = np.random.randint(1, 501, n_samples)
    job_categories = np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Marketing'], 
                                    n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    
    # Introduce gender bias in job categories
    tech_jobs = (job_categories == 'Tech')
    tech_bias = np.random.random(n_samples) < 0.3  # 30% chance of tech bias
    job_categories[tech_jobs & (user_genders == 1) & tech_bias] = 'Marketing'
    
    job_salaries = np.random.normal(75000, 25000, n_samples)
    job_required_experience = np.random.normal(3, 2, n_samples)
    job_company_sizes = np.random.choice(['Small', 'Medium', 'Large'], 
                                       n_samples, p=[0.4, 0.4, 0.2])
    
    # User-job compatibility
    user_skill_match = np.random.beta(2, 2, n_samples)  # Beta distribution for 0-1
    
    # Session features
    session_durations = np.random.exponential(300, n_samples)  # seconds
    session_positions = np.random.randint(1, 21, n_samples)
    
    # Click probability (introduce bias)
    base_click_prob = 0.3 + 0.4 * user_skill_match
    
    # Gender bias: reduce click probability for certain groups
    gender_bias = np.where(user_genders == 1, -0.1, 0.0)
    category_bias = np.where(job_categories == 'Tech', 0.05, 0.0)
    
    click_probability = np.clip(base_click_prob + gender_bias + category_bias, 0, 1)
    clicked = np.random.random(n_samples) < click_probability
    
    # Additional features for completeness
    job_locations = np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Austin'], 
                                   n_samples, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    
    job_types = np.random.choice(['Full-time', 'Part-time', 'Contract', 'Remote'], 
                               n_samples, p=[0.6, 0.15, 0.15, 0.1])
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'job_id': job_ids,
        'user_age': user_ages,
        'user_gender': user_genders,
        'user_education': user_educations,
        'user_experience_level': user_experience_levels,
        'user_skill_match': user_skill_match,
        'job_category': job_categories,
        'job_salary': job_salaries,
        'job_required_experience': job_required_experience,
        'job_company_size': job_company_sizes,
        'job_location': job_locations,
        'job_type': job_types,
        'session_duration': session_durations,
        'session_position': session_positions,
        'clicked': clicked.astype(int),
        'user_click_prob': click_probability,
        'job_popularity': np.random.beta(1, 3, n_samples),
        'user_job_history': np.random.poisson(5, n_samples),
        'time_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'user_activity_level': np.random.beta(2, 2, n_samples),
        'job_freshness': np.random.beta(3, 2, n_samples)
    })
    
    return df
