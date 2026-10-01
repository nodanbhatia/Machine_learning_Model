import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json
import os
import hashlib



st.set_page_config(page_title="Login", layout="wide")

USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)


def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title(" Social Media Opinion Mining")

    menu = st.sidebar.selectbox(
        "Menu",
        ["Login", "Register"]
    )

    users = load_users()

   

    if menu == "Login":

        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username in users and users[username] == hash_password(password):
                st.success("Login Successful")
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Username or Password")

   

    else:

        st.subheader("Create New Account")

        new_user = st.text_input("Username")

        new_pass = st.text_input("Password", type="password")

        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Register"):

            if new_user == "" or new_pass == "":
                st.warning("Please fill all fields.")

            elif new_user in users:
                st.error("Username already exists.")

            elif new_pass != confirm:
                st.error("Passwords do not match.")

            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.success("Registration Successful! Please Login.")

    st.stop()


st.set_page_config(page_title="Social Media Opinion & Sentiment Miner", layout="wide")


st.title(" Social Media Opinion Mining & Engagement Analytics")
st.markdown("""
This AI-powered dashboard analyzes social media metrics to mine public opinions, 
predict engagement tiers, and visualize platform-wide trends.
""")



def load_data():
    # Load the uploaded dataset
    df = pd.read_csv("social_media_engagement_dataset.csv")
    
    
    low_thresh = df['engagement_rate'].quantile(0.33)
    high_thresh = df['engagement_rate'].quantile(0.66)
    
    def label_sentiment(rate):
        if rate <= low_thresh:
            return 'Negative/Low'
        elif rate <= high_thresh:
            return 'Neutral/Medium'
        else:
            return 'Positive/High'
            
    df['Sentiment_Label'] = df['engagement_rate'].apply(label_sentiment)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Please ensure 'social_media_engagement_dataset.csv' is in the same directory as this script.")
    st.stop()



def train_model(data):
   
    ml_df = data.copy()
    
    
    le_platform = LabelEncoder()
    le_post = LabelEncoder()
    
    ml_df['platform_encoded'] = le_platform.fit_transform(ml_df['platform'])
    ml_df['post_type_encoded'] = le_post.fit_transform(ml_df['post_type'])
    
    
    X = ml_df[['platform_encoded', 'post_type_encoded', 'post_length', 'views', 'likes', 'comments', 'shares']]
    y = ml_df['Sentiment_Label']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
   
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    return model, le_platform, le_post, acc

model, le_platform, le_post, accuracy = train_model(df)


st.sidebar.header(" Predict Post Sentiment Tier")

input_platform = st.sidebar.selectbox("Select Platform", df['platform'].unique())
input_post_type = st.sidebar.selectbox("Select Post Type", df['post_type'].unique())
input_length = st.sidebar.slider("Post Length (Characters)", int(df['post_length'].min()), int(df['post_length'].max()), int(df['post_length'].mean()))
input_views = st.sidebar.number_input("Expected Views", value=int(df['views'].mean()))
input_likes = st.sidebar.number_input("Expected Likes", value=int(df['likes'].mean()))
input_comments = st.sidebar.number_input("Expected Comments", value=int(df['comments'].mean()))
input_shares = st.sidebar.number_input("Expected Shares", value=int(df['shares'].mean()))


if st.sidebar.button("Analyze & Predict Opinion Tier"):
    
    plat_encoded = le_platform.transform([input_platform])[0]
    post_encoded = le_post.transform([input_post_type])[0]
    
    
    user_features = np.array([[plat_encoded, post_encoded, input_length, input_views, input_likes, input_comments, input_shares]])
    
   
    prediction = model.predict(user_features)[0]
    
    st.sidebar.success(f"Predicted Public Sentiment/Engagement: **{prediction}**")


col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Posts Analyzed", value=len(df))
with col2:
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")
with col3:
    st.metric(label="Top Platform", value=df.groupby('platform')['engagement_rate'].mean().idxmax())

st.markdown("---")


col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("💡 Overall Public Sentiment Breakdown")
    sentiment_counts = df['Sentiment_Label'].value_counts()
    
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    colors1 = ["#20b04b","#d23737","#2d75c2"]
    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=colors1)
    ax1.axis('equal')  
    plt.tight_layout()
    st.pyplot(fig1)

with col_chart2:
    st.subheader(" Sentiment Trends Across Platforms")
    
    
    platform_sentiment = pd.crosstab(df['platform'], df['Sentiment_Label'])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    platform_sentiment.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel("Platform")
    ax2.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")


st.subheader(" Engagement Features vs Public Opinion")
feature_choice = st.selectbox("Select metric to compare against Public Reception:", ['likes', 'comments', 'shares', 'views'])


fig3, ax3 = plt.subplots(figsize=(10, 4))
categories = df['Sentiment_Label'].unique()
data_to_plot = [df[df['Sentiment_Label'] == cat][feature_choice].dropna() for cat in categories]

ax3.boxplot(data_to_plot, label=categories)
ax3.set_xticklabels(categories)
ax3.set_title(f"Distribution of {feature_choice.capitalize()} by Public Sentiment Class")
ax3.set_ylabel(feature_choice.capitalize())
plt.tight_layout()
st.pyplot(fig3)


st.markdown("---")
st.subheader(" Raw Data Preview with Mined Sentiments")
st.dataframe(df.head(10), use_container_width=True)