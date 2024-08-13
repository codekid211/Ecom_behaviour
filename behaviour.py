import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Customer Behavior Prediction", page_icon="🛒", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    logging.info("Loading data from CSV file")
    data = pd.read_csv('dataset/2019-Oct.csv')
    logging.info("Data loaded successfully with {} records".format(len(data)))
    return data

data = load_data()

def plot_category_distribution(data):
    logging.info("Plotting top 10 product categories")
    category_counts = data['category_code'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    plt.title('Top 10 Product Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_event_distribution(data):
    logging.info("Plotting event type distribution")
    event_counts = data['event_type'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
    plt.title('Event Type Distribution')
    return fig

def plot_hourly_activity(data):
    logging.info("Plotting hourly user activity")
    data['hour'] = pd.to_datetime(data['event_time']).dt.hour
    hourly_activity = data.groupby('hour')['event_type'].count()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=hourly_activity.index, y=hourly_activity.values, ax=ax)
    plt.title('Hourly User Activity')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Events')
    return fig

def prepare_data_for_model(data):
    logging.info("Preparing data for model")
    data['hour'] = pd.to_datetime(data['event_time']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['event_time']).dt.dayofweek
    data['is_purchase'] = (data['event_type'] == 'purchase').astype(int)
    features = ['hour', 'day_of_week', 'price']
    X = data[features]
    y = data['is_purchase']
    logging.info("Data preparation complete")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete")
    return model

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Model evaluation complete with accuracy: {:.2f}".format(accuracy))
    return accuracy, cm

def main():
    logging.info("Starting application")
    st.title("🛒 Customer Behavior Prediction")
    st.write("Analyze and predict customer behavior based on e-commerce data.")

    data = load_data()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Visualizations", "Predictive Analytics"])

    if page == "Data Overview":
        st.header("Data Overview")
        st.write(data.head())
        st.write(f"Total records: {len(data)}")
        st.write(f"Columns: {', '.join(data.columns)}")

    elif page == "Visualizations":
        st.header("Data Visualizations")
        
        st.subheader("Top 10 Product Categories")
        st.pyplot(plot_category_distribution(data))
        
        st.subheader("Event Type Distribution")
        st.pyplot(plot_event_distribution(data))
        
        st.subheader("Hourly User Activity")
        st.pyplot(plot_hourly_activity(data))

    elif page == "Predictive Analytics":
        st.header("Predictive Analytics")
        
        X_train, X_test, y_train, y_test = prepare_data_for_model(data)
        model = train_model(X_train, y_train)
        accuracy, cm = evaluate_model(model, X_test, y_test)
        
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        st.subheader("Make a Prediction")
        hour = st.slider("Hour of the day", 0, 23, 12)
        day = st.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        price = st.number_input("Price", min_value=0.0, value=50.0)
        
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        day_num = day_mapping[day]
        
        prediction = model.predict([[hour, day_num, price]])
        st.write("Prediction:", "Purchase" if prediction[0] == 1 else "No Purchase")

if __name__ == "__main__":
    main()
