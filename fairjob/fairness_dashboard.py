import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Fairness-Aware Job Recommender", layout="centered")
st.title("📊 Fairness-Aware Job Recommender Dashboard")

# --- Load training history ---
try:
    history = pd.read_csv("training_history.csv")
except FileNotFoundError:
    st.error("❌ 'training_history.csv' not found! Please place it in the same folder.")
    st.stop()

# --- 1️⃣ Training vs Validation Loss ---
st.subheader("📉 Training vs Validation Loss")
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
ax1.plot(history["val_loss"], label="Validation Loss", linewidth=2)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig1)

# --- 2️⃣ Fairness Metrics Over Epochs ---
st.subheader("⚖️ Fairness Metrics Over Epochs")
metrics = ["demographic_parity", "equalized_odds", "exposure_gap", "adversary_accuracy"]

fig2, ax2 = plt.subplots(figsize=(7, 4))
for m in metrics:
    ax2.plot(history[m], label=m.replace("_", " ").title(), linewidth=2)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Metric Value")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)
st.pyplot(fig2)

# --- 3️⃣ Final Fairness Metrics Table ---
st.subheader("🏁 Final Fairness Metrics Summary")

final_metrics = {
    "Demographic Parity": history["demographic_parity"].iloc[-1],
    "Equalized Odds": history["equalized_odds"].iloc[-1],
    "Exposure Gap": history["exposure_gap"].iloc[-1],
    "Adversary Accuracy": history["adversary_accuracy"].iloc[-1],
}

final_df = pd.DataFrame(list(final_metrics.items()), columns=["Metric", "Final Value"])
final_df["Final Value"] = final_df["Final Value"].apply(lambda x: f"{x:.6f}")
st.table(final_df)

st.success("✅ Dashboard loaded successfully.")
