# AI Employee - Data Analysis and Reporting System

## Overview

This project provides an interactive web application for performing data analysis and generating reports. The application allows users to upload datasets in CSV, Excel, or JSON formats, clean the data, and perform various analyses such as trend analysis, linear regression, K-Means clustering, and decision tree classification. The results are displayed in an intuitive and visually appealing manner using Streamlit.

## Features

- **File Upload**: Supports CSV, Excel, and JSON file formats.
- **Data Cleaning**: Automatically drops rows with missing values and retains numeric columns for analysis.
- **Interactive Analysis**:
  - Trend Analysis: Displays a correlation matrix for numeric data.
  - Linear Regression: Performs regression analysis and visualizes results.
  - K-Means Clustering: Groups data into clusters and displays the results.
  - Decision Tree: Classifies data and reports accuracy.
- **Comprehensive Reporting**: Generates a full report including all analyses and visualizations.
- **User-Friendly Interface**: Simple and intuitive UI powered by Streamlit.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   - Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

3. **Upload a File**:
   - Click on the file uploader and select a CSV, Excel, or JSON file.

4. **Perform Analysis**:
   - Enter queries like "linear regression", "K-Means clustering", "decision tree", "summary", or "trend analysis" to trigger specific analyses.
   - View the results directly in the web interface.

## Example Queries

- "Show trend analysis"
- "Perform linear regression"
- "Cluster the data using K-Means"
- "Generate a full report"

## Project Structure

- **`app.py`**: Main Streamlit application script.
- **`data_processing.py`**: Handles data loading and cleaning.
- **`report_generation.py`**: Performs data analysis and generates reports.
- **`requirements.txt`**: Lists all required Python dependencies.

## Dependencies

- Python 3.x
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Troubleshooting

- **File Upload Errors**: Ensure the file format is supported (CSV, Excel, JSON).
- **Visualization Issues**: Verify that all dependencies are installed correctly.
- **Query Handling**: Use simple and clear queries for best results.

## Future Improvements

- Support for additional file formats (e.g., XML, Parquet).
- Enhanced natural language processing for query handling.
- Integration of generative AI for automated insights.
- Interactive and dynamic visualizations.

