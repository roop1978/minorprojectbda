#!/usr/bin/env python3
"""
Test script to verify the trained model can make predictions
"""

import torch
import pandas as pd
import numpy as np
from models_simple import SimpleFairnessAwareJobRecommender as FairnessAwareJobRecommender, get_model_config
from utils import create_batch_data
import utils  # required for custom preprocessing and fairness classes


def test_model():
    """Test the trained model"""

    print("Testing trained model...")

    try:
        # ✅ Allowlist custom classes for safe deserialization
        torch.serialization.add_safe_globals([
            utils.DataPreprocessor,
            utils.FairnessMetrics
        ])

        # ✅ Explicitly set weights_only=False (safe since it's your model)
        checkpoint = torch.load("model.pt", map_location="cpu", weights_only=False)

        config = checkpoint["config"]
        preprocessor = checkpoint["preprocessor"]

        model = FairnessAwareJobRecommender(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("[SUCCESS] Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ✅ Load test data
        df = pd.read_csv("test_data.csv")
        print(f"[SUCCESS] Test data loaded: {df.shape}")

        # Take a small sample for testing
        test_sample = df.head(5)
        print(f"Testing on {len(test_sample)} samples...")

        # Preprocess test data
        df_processed = preprocessor.transform(test_sample)

        # Create batch data
        batch_data = create_batch_data(df_processed, preprocessor, "cpu")

        # Make predictions
        with torch.no_grad():
            outputs = model(batch_data, graph=None)
            predictions = outputs["click_prob"].squeeze().numpy()

        print("\n[PREDICTIONS] Results:")
        for i, (idx, row) in enumerate(test_sample.iterrows()):
            print(f"Sample {i + 1}:")
            print(f"  User ID: {row['user_id']}, Job ID: {row['job_id']}")
            print(f"  Gender: {'Female' if row['user_gender'] else 'Male'}")
            print(f"  Actual Click: {row['clicked']}")
            print(f"  Predicted Click Probability: {predictions[i]:.4f}")
            print(f"  Recommendation: {'Yes' if predictions[i] > 0.5 else 'No'}")
            print()

        # ✅ Fairness metrics
        fairness_metrics = utils.FairnessMetrics()

        sensitive_attr = torch.FloatTensor(test_sample["user_gender"].values)
        predictions_tensor = torch.FloatTensor(predictions)
        actual_clicks = torch.FloatTensor(test_sample["clicked"].values)

        demo_parity, prob_0, prob_1 = fairness_metrics.demographic_parity(predictions_tensor, sensitive_attr)
        eq_odds, tpr_0, tpr_1, fpr_0, fpr_1 = fairness_metrics.equalized_odds(actual_clicks, predictions_tensor,
                                                                              sensitive_attr)
        exposure_gap, exp_0, exp_1 = fairness_metrics.exposure_gap(predictions_tensor, sensitive_attr)
        adv_accuracy = fairness_metrics.adversary_accuracy(predictions_tensor, sensitive_attr)

        print("[FAIRNESS] Metrics on Test Sample:")
        print(f"  Demographic Parity Gap: {demo_parity:.4f}")
        print(f"  Equalized Odds Gap: {eq_odds:.4f}")
        print(f"  Exposure Gap: {exposure_gap:.4f}")
        print(f"  Adversary Accuracy: {adv_accuracy:.4f}")

        print("\n[SUCCESS] Model testing completed successfully!")

    except Exception as e:
        print(f"[ERROR] Error testing model: {str(e)}")
        return False

    return True


if __name__ == "__main__":
    test_model()
