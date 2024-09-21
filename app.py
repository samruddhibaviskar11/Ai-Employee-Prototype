import streamlit as st
import pandas as pd
import report_generation as rg
import data_processing as dp
import re
import seaborn as sns

# Upload file
st.title("AI Employee - Data Analysis and Reporting")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Load the data based on file type
    try:
        df = dp.load_data(uploaded_file)
        df_cleaned = dp.clean_data(df)

        # Display cleaned data
        st.write("Cleaned Data:")
        st.write(df_cleaned)
        st.write("Columns name:")
        st.write(df.columns)

        # Trend analysis
        st.write("Trend Analysis:")
        try:
            trends = rg.trend_analysis(df_cleaned)
            st.write("Correlation Matrix:")
            st.write(trends)
        except ValueError as e:
            st.error(f"Error during trend analysis: {e}")

        st.write("Ask the AI Employee about the analysis:")
        st.write("Ask for Trend Analysis, Full Summary, or report")
        st.write("Or perform linear regression, provide K-Means or clustering solutions, or perform decision tree analysis:")
        user_query = st.text_input("Enter your query")

        if user_query:
            # Basic NLP and if
            user_query = user_query.lower()

            if re.search(r"linear regression", user_query):
                st.write("Linear Regression Analysis:")
                rg.linear_regression_analysis(df_cleaned)
            
            elif re.search(r"cluster|kmeans|k-means|clustering", user_query):
                st.write("K-Means Cluster Analysis:")
                rg.kmeans_clustering(df_cleaned)
            
            elif re.search(r"decision tree", user_query):
                st.write("Decision Tree Classifier Results:")
                rg.decision_tree_analysis(df_cleaned)

            elif re.search(r"summary|report", user_query):
                st.write("Generating Full Summary Report:")
                rg.generate_report(df_cleaned)

            elif re.search(r"trend analysis", user_query):
                st.write("Trend Analysis using Correlation Matrix in details:")
                st.write(df_cleaned.describe())
                rg.trend_analysis(df_cleaned)
                st.write("hi")

            else:
                st.write("Sorry, I didn't understand your query. Please ask about linear regression, clustering, decision trees, summary, or report.")
    
    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
