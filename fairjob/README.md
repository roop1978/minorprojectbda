# 🎯 Fairness-Aware Job Recommendation System

A comprehensive end-to-end system that implements a hybrid **Wide+Deep+Session-GNN** model for job recommendations while actively reducing gender bias through adversarial debiasing and fairness regularization.

## 🏗️ Architecture Overview

The system combines four key components:

1. **Wide Component**: Memorizes sparse feature interactions
2. **Deep Component**: Learns complex feature representations  
3. **Session-GNN**: Captures user-job interaction patterns using Graph Neural Networks
4. **Adversary**: Detects and mitigates gender bias through adversarial training

### Loss Function
```
L_total = L_click - λ_adv × L_adv + λ_fair × L_fair
```

Where:
- `L_click`: Main click prediction loss
- `L_adv`: Adversarial loss (penalized to reduce bias detection)
- `L_fair`: Fairness regularization loss

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd fairjob/

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate default dataset (10,000 samples)
python generate_data.py

# Generate custom dataset
python generate_data.py --samples 20000 --output my_dataset.csv --seed 123
```

### 3. Train the Model

```bash
# Train with default parameters
python train.py --train

# Train with custom parameters
python train.py --train --epochs 100 --batch_size 512 --learning_rate 0.001
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

## 📊 Dataset Format

The system expects CSV data with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique user identifier |
| `job_id` | int | Unique job identifier |
| `user_age` | int | User's age |
| `user_gender` | int | Gender (0: male, 1: female) |
| `user_education` | str | Education level |
| `user_experience_level` | str | Experience level |
| `user_skill_match` | float | Skill-job match score (0-1) |
| `job_category` | str | Job category |
| `job_salary` | float | Job salary |
| `job_required_experience` | float | Required experience |
| `job_company_size` | str | Company size |
| `job_location` | str | Job location |
| `job_type` | str | Job type |
| `session_duration` | float | Session duration in seconds |
| `session_position` | int | Position in session |
| `clicked` | int | Whether user clicked (0/1) |

## ⚖️ Fairness Metrics

The system tracks several fairness metrics:

### Target Thresholds
- **Demographic Parity Gap**: ≤ 0.05
- **Equalized Odds Gap**: ≤ 0.05  
- **Exposure Gap**: ≤ 0.1
- **Adversary Accuracy**: < 0.6

### Metrics Explained

1. **Demographic Parity**: Equal positive prediction rates across gender groups
2. **Equalized Odds**: Equal true/false positive rates across groups
3. **Exposure Gap**: Equal recommendation exposure across groups
4. **Adversary Accuracy**: Measure of bias detection capability

## 🎛️ Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 256 | Training batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--lambda_adv` | 1.0 | Adversarial loss weight |
| `--lambda_fair` | 0.1 | Fairness loss weight |
| `--patience` | 15 | Early stopping patience |

## 📈 Dashboard Features

The Streamlit dashboard provides:

1. **Upload & Predict**: Upload CSV data and get click probability predictions
2. **Fairness Analysis**: Interactive plots of fairness metrics over training
3. **Model Performance**: Training curves and architecture information
4. **About**: Detailed system documentation

## 🔧 Model Configuration

The model architecture can be customized in `models.py`:

```python
config = {
    'num_wide_features': 10,
    'num_deep_features': 20,
    'user_dim': 15,
    'job_dim': 12,
    'embedding_dim': 32,
    'deep_hidden_dims': [256, 128, 64],
    'gnn_hidden_dim': 64,
    'gnn_num_layers': 2,
    'adversary_hidden_dims': [64, 32],
    'lambda_adv': 1.0,
    'lambda_fair': 0.1
}
```

## 📁 Project Structure

```
fairjob/
├── models.py          # Model architecture definitions
├── utils.py           # Fairness metrics and utilities
├── train.py           # Training script
├── app.py             # Streamlit dashboard
├── generate_data.py   # Synthetic data generation
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## 🎯 Example Usage

### Training with Custom Data

```bash
# 1. Prepare your data in the expected format
# 2. Train the model
python train.py --train --data your_data.csv --epochs 100

# 3. Launch dashboard
streamlit run app.py
```

### Making Predictions

```python
import torch
import pandas as pd
from models import FairnessAwareJobRecommender
from utils import DataPreprocessor, create_batch_data

# Load trained model
checkpoint = torch.load('model.pt', map_location='cpu')
model = FairnessAwareJobRecommender(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
preprocessor = checkpoint['preprocessor']

# Load new data
df = pd.read_csv('new_data.csv')
df_processed = preprocessor.transform(df)

# Make predictions
batch_data = create_batch_data(df_processed, preprocessor, 'cpu')
with torch.no_grad():
    outputs = model(batch_data)
    predictions = outputs['click_prob'].squeeze().numpy()
```

## 🔬 Research Background

This implementation is based on research in:

- **Fairness in Machine Learning**: Demographic parity, equalized odds
- **Adversarial Debiasing**: Using adversarial networks to reduce bias
- **Graph Neural Networks**: For capturing collaborative filtering patterns
- **Wide & Deep Learning**: Combining memorization and generalization

## 🛠️ Dependencies

- **PyTorch**: Deep learning framework
- **DGL**: Graph neural network library
- **Streamlit**: Web dashboard framework
- **Plotly**: Interactive visualizations
- **scikit-learn**: Data preprocessing
- **pandas/numpy**: Data manipulation

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or issues, please open an issue in the repository or contact the maintainers.

---

**Built with ❤️ for fair AI in job recommendations**
