import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_pandas_profiling import st_profile_report

from ydata_profiling import ProfileReport
from pycaret.classification import setup, compare_models, pull, save_model, plot_model
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AutoML", page_icon="🤖", layout="wide")

if "data" not in st.session_state:
    st.session_state.data = None

# Sidebar
with st.sidebar:
    st.title("BestModel")
    st.info(
        "The most accurate Machine Learning model for your dataset is now just a click away!"
    )
    choice = st.radio("Navigation", ["Upload", "Profiling", "Models", "Download"])

    with st.expander("About"):
        st.write(
            """
        This app helps you analyze your dataset and find the best machine learning model automatically.
        
        1. Upload your CSV file
        2. Explore data through profiling
        3. Train models to find the best one
        4. Download the trained model
        """
        )

# Upload page
if choice == "Upload":
    st.title("Upload your Data for Modelling")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            delimiter_option = st.radio(
                "Select CSV delimiter (if auto-detection fails)",
                options=[";", ",", "Auto-detect"],
                index=2,
            )

            if delimiter_option == "Auto-detect":
                sample_data = uploaded_file.read(1024)
                uploaded_file.seek(0)

                sample_str = sample_data.decode("utf-8", errors="replace")
                comma_count = sample_str.count(",")
                semicolon_count = sample_str.count(";")

                if semicolon_count > comma_count:
                    delimiter = ";"
                else:
                    delimiter = ","

                st.info(f"Auto-detected delimiter: '{delimiter}'")
            else:
                delimiter = delimiter_option

            df = pd.read_csv(uploaded_file, index_col=None, delimiter=delimiter)

            st.session_state.data = df

            df.to_csv("sourcedata.csv", index=None)

            # Show data summary
            st.success(f"✅ Successfully uploaded: {uploaded_file.name}")

            # Display sample of the data
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            st.subheader("Dataset Info")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Rows: {df.shape[0]}")
            with col2:
                st.info(f"Columns: {df.shape[1]}")

            # Show column types
            st.subheader("Column Data Types")
            dtypes = pd.DataFrame(df.dtypes, columns=["Data Type"])
            dtypes.index.name = "Column"
            st.dataframe(dtypes)

        except Exception as e:
            st.error(f"Error: {e}")

# Profiling page
elif choice == "Profiling":
    st.title("Automated Data Analysis")

    # Get data
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)
        st.session_state.data = df

    if st.session_state.data is None:
        st.error("Please upload a dataset first.")
    else:
        df = st.session_state.data

        # Create tabs
        tab1, tab2, tab3 = st.tabs(
            ["Data Overview", "Column Analysis", "Correlation Analysis"]
        )

        with tab1:
            st.subheader("Dataset Overview")
            st.write("Generate a comprehensive profile report of your dataset")

            if st.button("Generate Profile Report"):
                with st.spinner("Generating profile report..."):
                    try:
                        profile = ProfileReport(
                            df,
                            title="Data Profiling Report",
                            explorative=True,
                            minimal=False,
                        )
                        st_profile_report(profile)
                    except Exception as e:
                        st.error(f"An error occurred while generating the report: {e}")

        with tab2:
            st.subheader("Column-wise Analysis")

            # Column selector
            selected_column = st.selectbox("Select a column to analyze:", df.columns)

            if selected_column:
                col_data = df[selected_column]

                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric column
                    st.write(f"**{selected_column}** is a numeric column")

                    # Statistics
                    stats = col_data.describe().to_frame().T
                    st.dataframe(stats)

                    value_counts = col_data.value_counts().sort_index().reset_index()
                    value_counts.columns = [selected_column, "Count"]

                    # Create line chart with markers
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=value_counts[selected_column],
                            y=value_counts["Count"],
                            mode="lines+markers",
                            name=f"Distribution of {selected_column}",
                            line=dict(shape="spline", smoothing=1.3, width=2),
                            marker=dict(size=8),
                        )
                    )

                    fig.update_layout(
                        title=f"Distribution of {selected_column}",
                        xaxis_title=selected_column,
                        yaxis_title="Count",
                        hovermode="closest",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_data[
                        (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
                    ]
                    if not outliers.empty:
                        st.warning(
                            f"Detected {len(outliers)} potential outliers in {selected_column}"
                        )

                elif (
                    pd.api.types.is_categorical_dtype(col_data)
                    or col_data.nunique() < 20
                ):
                    # Categorical column
                    st.write(f"**{selected_column}** is a categorical column")

                    # Value counts
                    value_counts = col_data.value_counts().reset_index()
                    value_counts.columns = [selected_column, "Count"]

                    # Bar chart
                    fig = px.bar(
                        value_counts,
                        x=selected_column,
                        y="Count",
                        title=f"Distribution of {selected_column}",
                        color=selected_column,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(value_counts, use_container_width=True)

                else:
                    st.write(
                        f"**{selected_column}** is a text column or has high cardinality"
                    )

                    # Sample values
                    st.write("Sample values:")
                    st.dataframe(
                        col_data.sample(min(10, len(col_data))),
                        use_container_width=True,
                    )

                    # Statistics
                    n_unique = col_data.nunique()
                    n_missing = col_data.isna().sum()
                    stats_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Unique Values",
                                "Missing Values",
                                "Missing (%)",
                            ],
                            "Value": [
                                n_unique,
                                n_missing,
                                f"{n_missing/len(col_data)*100:.2f}%",
                            ],
                        }
                    )
                    st.dataframe(stats_df, use_container_width=True)

        with tab3:
            st.subheader("Correlation Analysis")

            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if len(numeric_cols) < 2:
                st.warning(
                    "Need at least two numeric columns for correlation analysis."
                )
            else:
                correlation = df[numeric_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                mask = np.triu(np.ones_like(correlation))
                sns.heatmap(
                    correlation,
                    mask=mask,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    ax=ax,
                )
                plt.title("Correlation Matrix")
                st.pyplot(fig)

                st.subheader("Correlation Table")
                st.dataframe(correlation.style.background_gradient(cmap="coolwarm"))

                st.subheader("Scatter Plot")
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox(
                        "Y-axis:", [c for c in numeric_cols if c != x_col], index=0
                    )

                if x_col != y_col:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{x_col} vs {y_col}",
                        trendline="ols",
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Models page
elif choice == "Models":
    st.title("Automated Machine Learning")

    # Get data
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col=None)
        st.session_state.data = df

    if st.session_state.data is None:
        st.error("Please upload a dataset first.")
    else:
        df = st.session_state.data

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Model Settings")

            # Target variable selector
            target = st.selectbox("Select target variable:", df.columns)

            # Advanced settings
            with st.expander("Advanced Settings"):
                # Preprocessing options
                preprocess = st.checkbox(
                    "Preprocess data (normalize, handle missing values)", value=True
                )
                remove_outliers = st.checkbox("Remove outliers", value=False)
                train_size = st.slider(
                    "Training data size (%)",
                    min_value=50,
                    max_value=90,
                    value=70,
                    step=5,
                )

                # Feature selection
                feature_selection = st.checkbox(
                    "Perform feature selection", value=False
                )

                # Model selection
                models_to_include = st.multiselect(
                    "Models to include:",
                    ["lr", "dt", "rf", "xgboost", "lightgbm", "catboost"],
                    default=["lr", "dt", "rf"],
                )

                # Model evaluation
                n_fold = st.slider(
                    "Cross-validation folds:", min_value=2, max_value=10, value=5
                )

        with col2:
            st.subheader("Train Model")

            if st.button("Train Models"):
                try:
                    st.info("Setting up experiment...")

                    setup_args = {
                        "data": df,
                        "target": target,
                        "train_size": train_size / 100,
                        "preprocess": preprocess,
                        "remove_outliers": remove_outliers,
                        "feature_selection": feature_selection,
                        "fold": n_fold,
                        "verbose": False,
                    }

                    exp = setup(**setup_args)

                    if models_to_include:
                        best_model = compare_models(include=models_to_include)
                    else:
                        best_model = compare_models()

                    setup_df = pull()
                    st.subheader("Experiment Settings")
                    st.dataframe(setup_df)

                    # Train models
                    st.info("Training and comparing models...")
                    best_model = compare_models()
                    compare_df = pull()

                    # Display results
                    st.subheader("Model Comparison")
                    st.dataframe(compare_df)

                    # Save the best model
                    save_model(best_model, "best_model")
                    st.success(
                        f"Best model trained and saved: {type(best_model).__name__}"
                    )

                    st.subheader("Model Visualizations")
                    tab1, tab2, tab3 = st.tabs(
                        ["Confusion Matrix", "ROC Curve", "Feature Importance"]
                    )

                    with tab1:
                        try:
                            cm_fig = plot_model(
                                best_model, plot="confusion_matrix", save=True
                            )
                            st.pyplot(cm_fig)
                        except:
                            st.info(
                                "Confusion matrix not available for this model type"
                            )

                    with tab2:
                        try:
                            roc_fig = plot_model(best_model, plot="auc", save=True)
                            st.pyplot(roc_fig)
                        except:
                            st.info("ROC curve not available for this model type")

                    with tab3:
                        try:
                            feat_fig = plot_model(best_model, plot="feature", save=True)
                            st.pyplot(feat_fig)
                        except:
                            st.info(
                                "Feature importance not available for this model type"
                            )

                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")

# Download page
elif choice == "Download":
    st.title("Download Trained Model")

    if os.path.exists("best_model.pkl"):
        st.success("✅ Model trained and ready for download")

        with open("best_model.pkl", "rb") as f:
            st.download_button(
                label="Download Model (.pkl)",
                data=f,
                file_name="trained_model.pkl",
                mime="application/octet-stream",
            )

        # Usage instructions
        with st.expander("How to use the downloaded model"):
            st.markdown(
                """
            #### Loading the model in Python:
            ```python
            import pickle
            
            # Load the model
            with open('trained_model.pkl', 'rb') as f:
                model = pickle.load(f)
                
            # Use the model to make predictions
            predictions = model.predict(X_test)
            ```
            """
            )
    else:
        st.error(
            "No trained model found. Please go to the Models page and train a model first."
        )
