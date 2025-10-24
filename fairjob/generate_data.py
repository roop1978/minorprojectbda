#!/usr/bin/env python3
"""
Script to generate synthetic FairJob dataset for testing the fairness-aware job recommendation system.
"""

import pandas as pd
import numpy as np
from utils import generate_synthetic_data
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic FairJob dataset')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='fairjob.csv', help='Output CSV filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print(f"Generating {args.samples} samples of synthetic FairJob data...")
    print(f"Random seed: {args.seed}")
    
    # Generate synthetic data
    df = generate_synthetic_data(args.samples)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"[SUCCESS] Synthetic data saved to {args.output}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic statistics
    print("\n[DATA] Dataset Statistics:")
    print(f"Gender distribution: {df['user_gender'].value_counts().to_dict()}")
    print(f"Click rate: {df['clicked'].mean():.3f}")
    print(f"Average age: {df['user_age'].mean():.1f}")
    print(f"Average salary: ${df['job_salary'].mean():,.0f}")
    
    # Check for bias
    male_click_rate = df[df['user_gender'] == 0]['clicked'].mean()
    female_click_rate = df[df['user_gender'] == 1]['clicked'].mean()
    bias_gap = male_click_rate - female_click_rate
    
    print(f"\n[BIAS] Bias Analysis:")
    print(f"Male click rate: {male_click_rate:.3f}")
    print(f"Female click rate: {female_click_rate:.3f}")
    print(f"Bias gap: {bias_gap:.3f}")
    
    if abs(bias_gap) > 0.05:
        print("[WARNING] Significant gender bias detected - good for testing fairness algorithms!")
    else:
        print("[INFO] Gender bias is minimal")


if __name__ == "__main__":
    main()
