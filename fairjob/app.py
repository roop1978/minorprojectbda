import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import utils
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

from models_simple_no_dgl import FairnessAwareJobRecommender, get_model_config
from utils import (
    DataPreprocessor, FairnessMetrics, 
    create_batch_data, create_interactive_fairness_dashboard
)


class FairnessAwareDashboard:
    """Streamlit dashboard for fairness-aware job recommendations"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.config = None
        self.fairness_metrics = FairnessMetrics()
        
        # Load model if available
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        model_path = "model_small.pt"  # use correct file name

        if os.path.exists(model_path):
            try:
                import torch
                import utils
                torch.serialization.add_safe_globals([
                    utils.DataPreprocessor,
                    utils.FairnessMetrics
                ])

                # ✅ Now safely load your checkpoint
                checkpoint = torch.load("model_small.pt", map_location=torch.device("cpu"), weights_only=False)

                # load config & preprocessor
                self.config = checkpoint.get("config", get_model_config())
                self.preprocessor = checkpoint.get("preprocessor", None)

                # initialize model and load weights
                self.model = FairnessAwareJobRecommender(self.config)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                st.success("✅ Model loaded successfully!")
                return True

            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                return False

        else:
            st.warning("⚠️ No trained model found. Please train first using: `python train.py --train`")
            return False

    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="Fairness-Aware Job Recommendation System",
            page_icon="🎯",
            layout="wide"
        )
        
        st.title("🎯 Fairness-Aware Job Recommendation System")
        st.markdown("---")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        if self.model is None:
            self.render_model_not_found()
        else:
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Upload & Predict", "📈 Fairness Analysis", "🎯 Model Performance", "ℹ️ About"])
            
            with tab1:
                self.render_prediction_tab()
            
            with tab2:
                self.render_fairness_analysis_tab()
            
            with tab3:
                self.render_model_performance_tab()
            
            with tab4:
                self.render_about_tab()
    
    def render_sidebar(self):
        """Render the sidebar"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Model status
        if self.model is not None:
            st.sidebar.success("✅ Model Ready")
            st.sidebar.info(f"**Model Architecture:** Wide+Deep+GNN+Adversary")
            st.sidebar.info(f"**Parameters:** {sum(p.numel() for p in self.model.parameters()):,}")
        else:
            st.sidebar.error("❌ Model Not Loaded")
        
        st.sidebar.markdown("---")
        
        # Fairness parameters
        st.sidebar.subheader("🎯 Fairness Metrics")
        st.sidebar.metric("Target Demographic Parity", "≤ 0.05", "Gap")
        st.sidebar.metric("Target Equalized Odds", "≤ 0.05", "Gap")
        st.sidebar.metric("Target Exposure Gap", "≤ 0.1", "Gap")
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.subheader("⚡ Quick Actions")
        if st.sidebar.button("🔄 Refresh Model"):
            self.load_model()
            st.rerun()
        
        if st.sidebar.button("📊 Generate Sample Data"):
            self.generate_sample_data()
    
    def render_model_not_found(self):
        """Render content when model is not found"""
        st.error("🚨 No trained model found!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 How to Train the Model")
            st.markdown("""
            1. **Generate synthetic data:**
               ```bash
               python train.py --train
               ```
            
            2. **Train with custom parameters:**
               ```bash
               python train.py --train --epochs 50 --batch_size 256
               ```
            
            3. **Check training progress:**
               - Model will be saved as `model.pt`
               - Training history saved as `training_history.csv`
               - Fairness metrics plot saved as `fairness_metrics.png`
            """)
        
        with col2:
            st.subheader("🎯 Model Architecture")
            st.markdown("""
            **Wide+Deep+Session-GNN Model:**
            
            - **Wide Component:** Memorizes feature interactions
            - **Deep Component:** Learns complex feature representations  
            - **Session-GNN:** Captures user-job interaction patterns
            - **Adversary:** Detects and mitigates gender bias
            
            **Fairness Loss:**
            ```
            L_total = L_click - λ_adv × L_adv + λ_fair × L_fair
            ```
            """)
    
    def render_prediction_tab(self):
        """Render the prediction tab"""
        st.header("📊 Upload Data & Get Predictions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file with job recommendation data",
            type=['csv'],
            help="Upload a CSV file with the same schema as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                
                # Display sample data
                with st.expander("📋 Sample Data", expanded=False):
                    st.dataframe(df.head(10))
                
                # Data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(df))
                with col2:
                    st.metric("Features", len(df.columns))
                with col3:
                    if 'user_gender' in df.columns:
                        gender_dist = df['user_gender'].value_counts()
                        st.metric("Gender Balance", f"{gender_dist.min()}/{gender_dist.max()}")
                
                # Preprocess data
                if st.button("🔮 Make Predictions", type="primary"):
                    self.make_predictions(df)
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        
        else:
            # Show sample data format
            self.show_sample_data_format()
    
    def make_predictions(self, df):
        """Make predictions on uploaded data"""
        try:
            with st.spinner("🔄 Making predictions..."):
                # Preprocess data
                df_processed = self.preprocessor.transform(df)
                
                # Create batch data
                batch_data = create_batch_data(df_processed, self.preprocessor, 'cpu')
                
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(batch_data)
                    predictions = outputs['click_prob'].squeeze().numpy()
                
                # Add predictions to dataframe
                df_with_predictions = df.copy()
                df_with_predictions['click_probability'] = predictions
                df_with_predictions['recommended'] = (predictions > 0.5).astype(int)
                
                # Display results
                st.success("✅ Predictions completed!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Click Probability", f"{predictions.mean():.3f}")
                with col2:
                    st.metric("Recommendations Made", f"{df_with_predictions['recommended'].sum()}/{len(df)}")
                with col3:
                    st.metric("Max Probability", f"{predictions.max():.3f}")
                with col4:
                    st.metric("Min Probability", f"{predictions.min():.3f}")
                
                # Display predictions
                st.subheader("📊 Prediction Results")
                
                # Sort by probability
                df_sorted = df_with_predictions.sort_values('click_probability', ascending=False)
                
                # Top recommendations
                st.subheader("🏆 Top Recommendations")
                top_recommendations = df_sorted.head(20)[['user_id', 'job_id', 'click_probability', 'recommended']]
                st.dataframe(top_recommendations)
                
                # Distribution plot
                fig = px.histogram(
                    df_with_predictions, 
                    x='click_probability',
                    title='Distribution of Click Probabilities',
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = df_with_predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="job_recommendations.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error making predictions: {str(e)}")
    
    def show_sample_data_format(self):
        """Show sample data format"""
        st.subheader("📋 Expected Data Format")
        
        sample_data = pd.DataFrame({
            'user_id': [1, 2, 3],
            'job_id': [101, 102, 103],
            'user_age': [28, 35, 42],
            'user_gender': [0, 1, 0],
            'user_education': ['Bachelor', 'Master', 'PhD'],
            'user_experience_level': ['Mid', 'Senior', 'Senior'],
            'user_skill_match': [0.8, 0.6, 0.9],
            'job_category': ['Tech', 'Finance', 'Tech'],
            'job_salary': [75000, 95000, 120000],
            'job_required_experience': [3, 5, 7],
            'job_company_size': ['Medium', 'Large', 'Large'],
            'job_location': ['San Francisco', 'New York', 'Seattle'],
            'job_type': ['Full-time', 'Full-time', 'Full-time'],
            'session_duration': [300, 450, 200],
            'session_position': [1, 3, 2]
        })
        
        st.dataframe(sample_data)
        st.info("💡 Upload a CSV file with similar columns to get predictions!")
    
    def render_fairness_analysis_tab(self):
        """Render the fairness analysis tab"""
        st.header("📈 Fairness Analysis")
        
        if not os.path.exists('training_history.csv'):
            st.warning("⚠️ No training history found. Please train the model first.")
            return
        
        # Load training history
        history_df = pd.read_csv('training_history.csv')
        
        # Create interactive plots
        fig = create_interactive_fairness_dashboard(history_df.to_dict('list'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Current fairness metrics
        st.subheader("🎯 Current Fairness Metrics")
        
        if len(history_df) > 0:
            latest_metrics = history_df.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                demo_parity = latest_metrics['demographic_parity']
                color = "green" if abs(demo_parity) <= 0.05 else "red"
                st.metric(
                    "Demographic Parity Gap", 
                    f"{demo_parity:.4f}",
                    delta=None,
                    delta_color="normal"
                )
                if abs(demo_parity) <= 0.05:
                    st.success("✅ Within acceptable range")
                else:
                    st.error("❌ Needs improvement")
            
            with col2:
                eq_odds = latest_metrics['equalized_odds']
                color = "green" if eq_odds <= 0.05 else "red"
                st.metric(
                    "Equalized Odds Gap",
                    f"{eq_odds:.4f}",
                    delta=None,
                    delta_color="normal"
                )
                if eq_odds <= 0.05:
                    st.success("✅ Within acceptable range")
                else:
                    st.error("❌ Needs improvement")
            
            with col3:
                exposure_gap = latest_metrics['exposure_gap']
                st.metric(
                    "Exposure Gap",
                    f"{exposure_gap:.4f}",
                    delta=None,
                    delta_color="normal"
                )
                if abs(exposure_gap) <= 0.1:
                    st.success("✅ Within acceptable range")
                else:
                    st.error("❌ Needs improvement")
            
            with col4:
                adv_accuracy = latest_metrics['adversary_accuracy']
                st.metric(
                    "Adversary Accuracy",
                    f"{adv_accuracy:.4f}",
                    delta=None,
                    delta_color="normal"
                )
                if adv_accuracy < 0.6:
                    st.success("✅ Good bias mitigation")
                else:
                    st.warning("⚠️ Adversary still detecting bias")
        
        # Fairness recommendations
        st.subheader("💡 Fairness Recommendations")
        
        recommendations = [
            "🎯 **Demographic Parity Gap ≤ 0.05:** Ensure equal positive prediction rates across gender groups",
            "⚖️ **Equalized Odds Gap ≤ 0.05:** Minimize differences in true/false positive rates",
            "📊 **Exposure Gap ≤ 0.1:** Reduce differences in recommendation exposure",
            "🛡️ **Adversary Accuracy < 0.6:** Keep adversary from easily predicting sensitive attributes"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    
    def render_model_performance_tab(self):
        """Render the model performance tab"""
        st.header("🎯 Model Performance")
        
        if not os.path.exists('training_history.csv'):
            st.warning("⚠️ No training history found. Please train the model first.")
            return
        
        # Load training history
        history_df = pd.read_csv('training_history.csv')
        
        # Training curves
        st.subheader("📊 Training Curves")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training vs Validation Loss', 'Loss Components'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history_df) + 1))
        
        # Training vs Validation Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history_df['train_loss'], 
                      name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=epochs, y=history_df['val_loss'], 
                      name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model architecture info
        st.subheader("🏗️ Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Wide Component:**
            - Linear transformations
            - Feature interaction embeddings
            - Memorizes sparse feature combinations
            
            **Deep Component:**
            - Multi-layer neural network
            - Batch normalization
            - Dropout regularization
            - Learns complex feature representations
            """)
        
        with col2:
            st.markdown("""
            **Session-GNN:**
            - Graph neural network for user-job interactions
            - Multi-head attention for session aggregation
            - Captures collaborative filtering patterns
            
            **Adversary:**
            - Spectral normalization
            - Detects gender bias in predictions
            - Enables adversarial debiasing
            """)
        
        # Model parameters
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            with col2:
                st.metric("Trainable Parameters", f"{trainable_params:,}")
            with col3:
                st.metric("Model Size", f"{total_params * 4 / 1024 / 1024:.2f} MB")
    
    def render_about_tab(self):
        """Render the about tab"""
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ## 🎯 Fairness-Aware Job Recommendation System
        
        This system implements a hybrid **Wide+Deep+Session-GNN** model that predicts job click probabilities 
        while actively reducing gender bias through adversarial debiasing and fairness regularization.
        
        ### 🏗️ Architecture Components
        
        #### 1. **Wide Component**
        - Memorizes sparse feature interactions
        - Linear transformations + embedding interactions
        - Captures memorizable patterns
        
        #### 2. **Deep Component** 
        - Multi-layer neural network with batch normalization
        - Dropout regularization for generalization
        - Learns complex feature representations
        
        #### 3. **Session-GNN Component**
        - Graph Neural Network for user-job interactions
        - Multi-head attention for session-level aggregation
        - Captures collaborative filtering patterns
        
        #### 4. **Adversarial Component**
        - Detects gender bias in predictions
        - Uses spectral normalization for stability
        - Enables adversarial debiasing
        
        ### ⚖️ Fairness Mechanisms
        
        #### **Adversarial Debiasing**
        ```
        L_total = L_click - λ_adv × L_adv + λ_fair × L_fair
        ```
        - **L_click:** Main prediction loss
        - **L_adv:** Adversarial loss (penalized to reduce bias detection)
        - **L_fair:** Fairness regularization
        
        #### **Fairness Metrics**
        - **Demographic Parity:** Equal positive prediction rates across groups
        - **Equalized Odds:** Equal true/false positive rates across groups  
        - **Exposure Gap:** Equal recommendation exposure across groups
        - **Adversary Accuracy:** Measure of bias detection capability
        
        ### 🎯 Target Fairness Thresholds
        
        - **Demographic Parity Gap:** ≤ 0.05
        - **Equalized Odds Gap:** ≤ 0.05
        - **Exposure Gap:** ≤ 0.1
        - **Adversary Accuracy:** < 0.6
        
        ### 🚀 Usage
        
        1. **Train the model:**
           ```bash
           python train.py --train --epochs 50
           ```
        
        2. **Launch dashboard:**
           ```bash
           streamlit run app.py
           ```
        
        3. **Upload data and get predictions with fairness analysis**
        
        ### 📚 Technical Details
        
        - **Framework:** PyTorch + DGL (Graph Neural Networks)
        - **Frontend:** Streamlit
        - **Visualization:** Plotly
        - **Fairness:** Adversarial debiasing + regularization
        
        ### 🔬 Research Background
        
        This implementation is based on research in:
        - Fairness in machine learning
        - Adversarial debiasing techniques
        - Graph neural networks for recommendations
        - Wide & Deep learning architectures
        """)
        
        # Contact info
        st.markdown("---")
        st.markdown("**Built with ❤️ for fair AI in job recommendations**")


def main():
    """Main function to run the dashboard"""
    dashboard = FairnessAwareDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
