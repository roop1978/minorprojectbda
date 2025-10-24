# 🚀 How to Use the Fairness-Aware Job Recommendation Dashboard

## 📋 Step-by-Step Guide

### 1. 🏃‍♂️ **Running the Dashboard**

```bash
# Navigate to the fairjob directory
cd fairjob

# Run the Streamlit dashboard
streamlit run app.py
```

The dashboard will open in your browser at: **http://localhost:8501**

### 2. 📊 **What to Upload in the Dashboard**

The dashboard expects CSV files with **exactly these columns**:

#### Required Columns:
```
user_id, job_id, user_age, user_gender, user_education, user_experience_level, 
user_skill_match, job_category, job_salary, job_required_experience, 
job_company_size, job_location, job_type, session_duration, session_position
```

#### Optional Columns (for advanced features):
```
clicked, user_click_prob, job_popularity, user_job_history, time_of_day, 
day_of_week, user_activity_level, job_freshness
```

### 3. 📁 **Sample Files to Upload**

I've created these files for you to test with:

- **`sample_data.csv`** - 50 samples (perfect for testing)
- **`test_data.csv`** - 1000 samples (larger dataset)

### 4. 🎯 **Dashboard Features**

#### **Tab 1: Upload & Predict**
- Upload your CSV file
- Click "Make Predictions" 
- View click probability predictions
- Download results as CSV

#### **Tab 2: Fairness Analysis**
- Interactive fairness metrics plots
- Current fairness thresholds
- Recommendations for improvement

#### **Tab 3: Model Performance**
- Training curves
- Model architecture details
- Parameter counts

#### **Tab 4: About**
- Complete system documentation
- Technical details
- Usage instructions

### 5. 📊 **Sample Data Format**

Here's what your CSV should look like:

```csv
user_id,job_id,user_age,user_gender,user_education,user_experience_level,user_skill_match,job_category,job_salary,job_required_experience,job_company_size,job_location,job_type,session_duration,session_position
1,101,28,0,Bachelor,Mid,0.8,Tech,75000,3,Medium,San Francisco,Full-time,300,1
2,102,35,1,Master,Senior,0.6,Finance,95000,5,Large,New York,Full-time,450,2
```

### 6. ⚖️ **Understanding Fairness Metrics**

- **Demographic Parity Gap**: ≤ 0.05 (Equal prediction rates across genders)
- **Equalized Odds Gap**: ≤ 0.05 (Equal true/false positive rates)
- **Exposure Gap**: ≤ 0.1 (Equal recommendation exposure)
- **Adversary Accuracy**: < 0.6 (Bias detection capability)

### 7. 🔧 **Troubleshooting**

#### If the dashboard doesn't load:
```bash
# Make sure you're in the right directory
cd fairjob

# Check if all files are present
ls

# Restart Streamlit
streamlit run app.py
```

#### If predictions fail:
- Check your CSV has all required columns
- Ensure data types are correct (numbers for numerical columns)
- Try with the sample_data.csv first

### 8. 🎯 **Quick Test**

1. Open the dashboard at http://localhost:8501
2. Go to "Upload & Predict" tab
3. Upload `sample_data.csv`
4. Click "Make Predictions"
5. View the results and fairness metrics!

## 🎉 **You're Ready to Go!**

The system is now running and ready to analyze job recommendation fairness. Upload your data and start exploring the predictions and fairness metrics!
