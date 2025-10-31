import pandas as pd
import matplotlib.pyplot as plt

# Load the training history CSV
history = pd.read_csv("training_history.csv")

# 1️⃣ Training vs Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 2️⃣ Fairness Metrics Over Epochs
metrics = ['demographic_parity', 'equalized_odds', 'exposure_gap', 'adversary_accuracy']
plt.figure(figsize=(10,6))
for metric in metrics:
    plt.plot(history[metric], label=metric.replace('_', ' ').title(), linewidth=2)
plt.title("Fairness Metrics Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# 3️⃣ Final Fairness Metrics Summary
final_metrics = {
    "Demographic Parity": history['demographic_parity'].iloc[-1],
    "Equalized Odds": history['equalized_odds'].iloc[-1],
    "Exposure Gap": history['exposure_gap'].iloc[-1],
    "Adversary Accuracy": history['adversary_accuracy'].iloc[-1]
}

plt.figure(figsize=(7,4))
plt.bar(final_metrics.keys(), final_metrics.values(),
        color=['#4CAF50','#FFC107','#2196F3','#9C27B0'])
plt.title("Final Fairness Metrics (Lower = Better)")
plt.ylabel("Metric Value")
plt.show()
