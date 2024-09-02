import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open('autoencoder.pkl', 'rb') as file:
    autoencoder_model = pickle.load(file)

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def classify_transactions(autoencoder_model, new_data):
    preprocessed_data = preprocess_data(new_data)
    reconstructions = autoencoder_model.predict(preprocessed_data)
    reconstruction_error = np.mean(np.square(preprocessed_data - reconstructions), axis=1)

    threshold = np.percentile(reconstruction_error, 95)  

    new_data['Fraudulent'] = reconstruction_error > threshold
    fraudulent_transactions = new_data[new_data['Fraudulent']]
    return fraudulent_transactions

def visualize_fraudulent_transactions(fraudulent_transactions):
    st.subheader('Fraudulent Transactions Visualization')
    feature_x = 'Amount'
    feature_y = 'Time'  
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=fraudulent_transactions, x=feature_x, y=feature_y,color='red', alpha=0.6)
    plt.title('Fraudulent Transactions')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    st.pyplot(plt)

def main():
    st.title('Credit Card Fraud Detection')

    st.sidebar.header('Upload New Dataset')
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        non_numeric_cols = new_data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            st.error(f"Error: The dataset contains non-numeric columns: {', '.join(non_numeric_cols)}. Please clean the data and upload again.")
        else:
            st.write("Data Preview", new_data.head())        
            numeric_data = new_data.select_dtypes(include=[np.number])
            fraudulent_transactions = classify_transactions(autoencoder_model, numeric_data)
            
            if not fraudulent_transactions.empty:
                st.write("Fraudulent Transactions Detected:")
                st.write(fraudulent_transactions)
                visualize_fraudulent_transactions(fraudulent_transactions)
            else:
                st.write("No fraudulent transactions detected.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
