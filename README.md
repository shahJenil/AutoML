# AutoML - BestModel Finder

## Overview

AutoML is an automated machine learning web application designed to simplify data analysis and model selection. Built with Streamlit, it allows users to upload a CSV dataset, perform exploratory data analysis (EDA), automatically train and compare machine learning models, and download the best-performing model.

### Key Features
1. **Data Upload**: Supports CSV files with automatic delimiter detection.
2. **Exploratory Data Analysis (EDA)**:
   - Comprehensive dataset profiling
   - Column-wise analysis (numeric, categorical, text)
   - Correlation analysis with heatmaps and scatter plots
3. **Automated Model Selection**:
   - Trains multiple classification/regression models
   - Compares performance metrics to select the best model
   - Customizable settings (target variable, train-test split, cross-validation)
4. **Model Evaluation**:
   - Visualizations: Confusion Matrix, ROC Curve, Feature Importance
5. **Model Export**: Download the best model in `.pkl` format.

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Profiling**: ydata-profiling, streamlit-pandas-profiling
- **Machine Learning**: PyCaret
- **File Handling**: Python `os` module

## Usage
- **Upload**: Go to the "Upload" section and upload your CSV dataset.
- **Profiling**: Explore your data with automated profiling, column analysis, and correlations under the "Profiling" tab.
- **Models**: Select a target variable (Only supports supervised-learning currently), adjust additional settings (e.g., train-test split, cross-validation folds), and click "Train Models" to find the best model.
- **Download**: Download the best fit trained model from the "Download" section as a .pkl file.
