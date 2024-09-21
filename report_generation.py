import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import streamlit as st

def data_summary(df):
    """Show basic statistics of the data"""
    st.write("Data Summary:")
    st.write(df.describe())
def trend_analysis(df):
    """Perform trend analysis and show the correlation matrix"""
    # Select only numeric columns 
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if numeric_df.empty:
        raise ValueError("No numeric data available for trend analysis.")
    
    # Perform correlation analysis
    correlation_matrix = numeric_df.corr()
    
    return correlation_matrix

def linear_regression_analysis(df):
    X = df.drop(columns=[df.columns[-1]])  # All columns except the last as features
    y = df[df.columns[-1]]  # Last column as target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # Scatter Plot with Regression Line
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.regplot(x=X.columns[0], y=df.columns[-1], data=df, ci=None, color='blue', line_kws={'color': 'red'})
    plt.title('Scatter Plot with Linear Regression Line')
    plt.xlabel(X.columns[0])
    plt.ylabel(df.columns[-1])
    
    # Show plot
    st.pyplot(plt)

    # Display Linear Regression Performance
    st.write(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred):.2f}")

def kmeans_clustering(df):
    """Perform K-Means Clustering and show the clusters"""
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(df)

    st.write("K-Means Clustering Results:")
    st.write(df['Cluster'])
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[df.columns[0]], y=df[df.columns[1]], hue='Cluster', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

def decision_tree_analysis(df):
    """Perform Decision Tree classification and accuracy"""
    X = df.drop(columns=[df.columns[-1]])  # All columns except the last as features
    y = df[df.columns[-1]]  # Last column as target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_tree = dt.predict(X_test)

    # Decision Tree Performance
    accuracy = accuracy_score(y_test, y_pred_tree)
    st.write(f"Decision Tree Accuracy: {accuracy:.2f}")



def generate_report(df):
    """Generate the full report including all analyses"""
    st.write("Data Summary:")
    st.write(df.describe())

    # Generate Correlation Matrix heatmap
    st.write("Correlation Matrix Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Box Plot of Dependent Variable
    
    # Box Plot of Dependent Variable
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 2)
    sns.boxplot(y=df[df.columns[-1]], color='blue')
    plt.title('Box Plot of Dependent Variable')
    plt.ylabel(df.columns[-1])
    
    # Display plots
    plt.tight_layout()
    st.pyplot(plt)
    # Linear Regression
    linear_regression_analysis(df)

    # K-Means Clustering
    kmeans_clustering(df)

    # Decision Tree
    decision_tree_analysis(df)

    st.write("Report Generation Complete.")
