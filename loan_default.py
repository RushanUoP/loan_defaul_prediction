import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay ,classification_report
import warnings
import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
# or
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore') 




def main():
    st.title("Loan Default Prediction System")
    df = pd.read_csv('Loan_Default.csv')
    st.write("First 5 rows of the data")
    st.write(df.head())
    st.write("Complete data")
    st.write(df)
    st.write("Columns of the data")
    st.write(df.columns)
    st.write("Information about the data")
    st.write(df.info())
    st.write("Description of the data")
    st.write(df.describe())
    st.write("Null values in the data")
    st.write(df.isnull().sum())
    st.write('Gender distribution')
    st.write(df['Gender'].value_counts())
    st.write('Age distribution')
    st.write(df['age'].value_counts())
    st.write('Null values in all columns')
    st.write(df.isnull().sum())

    columns_to_drop = ['loan_limit', 'rate_of_interest', 'Interest_rate_spread', 
                   'Upfront_charges', 'property_value', 'dtir1', 'LTV']
    df = df.drop(columns=columns_to_drop)
    st.write("Data after dropping columns")
    st.write(df)
    
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['float64', 'int64']:  
                df[column].fillna(df[column].median(), inplace=True)
            else:  
                df[column].fillna(df[column].mode()[0], inplace=True)
    st.write("Data after filling null values")
    st.write(df)

    label_columns = ['Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness', 'open_credit', 
                 'business_or_commercial', 'Neg_ammortization', 'interest_only', 'lump_sum_payment', 
                 'construction_type', 'occupancy_type', 'Secured_by', 'total_units', 'credit_type', 
                 'co-applicant_credit_type', 'age', 'submission_of_application', 'Region', 'Security_Type']
    for column in label_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    st.write("Data after encoding")
    st.write(df)
    st.write('Data Types')
    st.write(df.dtypes)

    corr = df.corr()
    st.write("Correlation Matrix")
    st.write(corr)

    # Display correlation matrix heatmap using Streamlit
    st.write("Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', cbar=True, ax=ax)
    plt.title('Correlation Matrix Heatmap')
    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create scatter plot using Plotly Express
    fig_scatter = px.scatter(df, x='income', y='Credit_Score', color='Status',
                    title='Income vs Credit Score',
                    labels={'income': 'Income', 'Credit_Score': 'Credit Score'})

    # Display the plot in Streamlit
    st.plotly_chart(fig_scatter)
    # Create count plot using seaborn
    fig_loan_status, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Status', data=df, ax=ax)
    plt.title('Loan Approval Status Distribution')

    # Display the plot in Streamlit
    st.pyplot(fig_loan_status)

        # Create boxplots for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)
    num_rows = (num_columns // 5) + (num_columns % 5 > 0)
    num_cols = 5

    # Create subplot
    fig_outliers, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))
    axes = axes.ravel()  # Flatten the axes array for easier indexing

    # Create boxplots
    for i, column in enumerate(numeric_columns):
        sns.boxplot(y=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')

    # Remove empty subplots if any
    for i in range(len(numeric_columns), len(axes)):
        fig_outliers.delaxes(axes[i])

    plt.tight_layout()

    # Display in Streamlit
    #st.write("Boxplots of Numeric Features")
    #st.pyplot(fig_outliers)

        # Create scaler
    scaler = RobustScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Show data before scaling
    st.write("Sample of numeric data before scaling:")
    st.write(df[numeric_cols].head())

    # Apply scaling
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Show data after scaling
    st.write("Sample of numeric data after scaling:")
    st.write(df[numeric_cols].head())


        # Add this after your data preprocessing steps
    st.write("## Model Training and Predictions")

    # Add a slider for threshold adjustment
    threshold = st.slider('Select Probability Threshold', 0.0, 1.0, 0.4, 0.05)

    # Create a button to run predictions
    if st.button('Run Model Predictions'):
        # Split the data

        #columns_to_keep = ['income', 'credit_type', 'Credit_Score', 'loan_amount', 'Status']
        #df = df[columns_to_keep]
        X = df.drop('Status', axis=1)  
        y = df['Status']

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
        
        # Create and train the model
        with st.spinner('Training Random Forest model...'):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
        
        # Make predictions with probability and threshold adjustment
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
        
        # Display results
        st.write("### Model Performance Metrics")
        
        # Display accuracy with threshold adjustment
        accuracy = accuracy_score(y_test, y_pred_adjusted)
        st.write(f"Random Forest Accuracy (with threshold {threshold}): {accuracy:.4f}")
        
        # Display classification report
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred_adjusted)
        st.text(report)
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_adjusted, ax=ax)
        plt.title(f'Confusion Matrix (threshold: {threshold})')
        st.pyplot(fig)
        
        # Display ROC curve
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig = px.line(x=fpr, y=tpr, 
                    title=f'ROC Curve (threshold: {threshold})',
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        fig.add_shape(type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)
        
        # Display feature importance
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance, x='feature', y='importance',
                    title='Feature Importance Plot')
        st.plotly_chart(fig)
if __name__ == "__main__":
    main()
