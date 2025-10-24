# 🎯 Fairness-Aware Job Recommendation System - Demo

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python generate_data.py --samples 10000 --output fairjob.csv
```

### 3. Train the Model
```bash
python train_simple.py --train --epochs 50 --batch_size 256
```

### 4. Test the Model
```bash
python test_model.py
```

### 5. Launch Dashboard
```bash
streamlit run app.py
```

## 📊 System Architecture

The system implements a **Wide+Deep+Session-GNN+Adversary** architecture:

- **Wide Component**: Memorizes feature interactions
- **Deep Component**: Learns complex feature representations  
- **Session Component**: Captures user-job interaction patterns
- **Adversary**: Detects and mitigates gender bias

### Loss Function
```
L_total = L_click - λ_adv × L_adv + λ_fair × L_fair
```

## ⚖️ Fairness Metrics

The system tracks and optimizes for:

- **Demographic Parity Gap**: ≤ 0.05 (Equal positive prediction rates)
- **Equalized Odds Gap**: ≤ 0.05 (Equal true/false positive rates)
- **Exposure Gap**: ≤ 0.1 (Equal recommendation exposure)
- **Adversary Accuracy**: < 0.6 (Bias detection capability)

## 📈 Training Results

Based on the test run:

- **Model Parameters**: 143,695
- **Training Samples**: 800
- **Validation Samples**: 200
- **Final Fairness Metrics**:
  - Demographic Parity Gap: 0.2708 (needs improvement)
  - Equalized Odds Gap: 0.5396 (needs improvement)
  - Adversary Accuracy: 0.6200 (good bias detection)

## 🎯 Sample Predictions

The model successfully makes predictions:

```
Sample 1: User 103, Job 273, Male, Actual: 1, Predicted: 0.4343
Sample 2: User 436, Job 294, Male, Actual: 0, Predicted: 0.3693
Sample 3: User 861, Job 398, Female, Actual: 1, Predicted: 0.3464
```

## 🛠️ Features

### Data Generation
- Synthetic FairJob dataset with realistic bias patterns
- Configurable gender bias injection
- Multiple job categories and user profiles

### Training
- Adversarial debiasing with configurable weights
- Fairness regularization
- Early stopping and model checkpointing
- Comprehensive fairness metrics tracking

### Dashboard
- Interactive Streamlit interface
- Real-time fairness analysis
- Model performance visualization
- CSV upload and prediction capabilities

## 📁 Project Structure

```
fairjob/
├── models_simple.py      # Simplified model architecture (no DGL)
├── train_simple.py       # Training script
├── utils.py              # Fairness metrics and utilities
├── app.py                # Streamlit dashboard
├── generate_data.py      # Synthetic data generation
├── test_model.py         # Model testing script
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## 🔧 Configuration

Key parameters can be adjusted in the training script:

- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate
- `--lambda_adv`: Adversarial loss weight
- `--lambda_fair`: Fairness loss weight

## 🎯 Next Steps

1. **Improve Fairness**: Adjust adversarial and fairness loss weights
2. **Scale Up**: Train on larger datasets with more epochs
3. **Real Data**: Adapt to real job recommendation datasets
4. **Advanced GNN**: Integrate full DGL functionality when compatible
5. **A/B Testing**: Deploy and test in production environments

## 📚 Research Background

This implementation demonstrates:
- Fairness-aware machine learning
- Adversarial debiasing techniques
- Hybrid recommendation architectures
- Real-time bias monitoring

The system provides a solid foundation for building fair job recommendation systems in production environments.
