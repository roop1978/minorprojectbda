import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="FairJob Model Dashboard", layout="wide")

st.title("📊 FairJob: Model Training & Fairness Visualization")

# Load CSV
history = pd.read_csv("training_history.csv")
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose what to view:", ["Training Progress", "Fairness Metrics", "Final Summary"])

# 1️⃣ Training Progress
if option == "Training Progress":
    st.subheader("Training vs Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

# 2️⃣ Fairness Metrics
elif option == "Fairness Metrics":
    st.subheader("Fairness Metrics Over Epochs")
    metrics = ['demographic_parity', 'equalized_odds', 'exposure_gap', 'adversary_accuracy']
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.plot(history[metric], label=metric.replace('_', ' ').title(), linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

# 3️⃣ Final Summary
else:
    st.subheader("Final Fairness Summary")
    final_metrics = {
        "Demographic Parity": history['demographic_parity'].iloc[-1],
        "Equalized Odds": history['equalized_odds'].iloc[-1],
        "Exposure Gap": history['exposure_gap'].iloc[-1],
        "Adversary Accuracy": history['adversary_accuracy'].iloc[-1]
    }
    st.dataframe(pd.DataFrame(final_metrics, index=["Final Values"]))
