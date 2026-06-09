import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

# -----------------------
# Load FAQ data
# -----------------------
data = pd.read_csv("faq.csv")
questions = data["question"].tolist()
answers = data["answer"].tolist()

# -----------------------
# NLP Model
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_input = user_input.lower()

    # 🔥 Custom logic (IMPORTANT FIX)
    if "my name is" in user_input:
        name = user_input.replace("my name is", "").strip()
        return f"Nice to meet you, {name}! 😊"

    if "what is my name" in user_input:
        return "You told me your name earlier 😊"

    # NLP Matching
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.5:
        return "Sorry, I didn't understand that. Please ask something else."

    return answers[index]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("🤖 AI FAQ Chatbot - CodeAlpha Internship Project")
st.markdown("### Built using NLP (TF-IDF + Cosine Similarity)")

# -----------------------
# Session state
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "saved" not in st.session_state:
    st.session_state.saved = False

# -----------------------
# User details
# -----------------------
name = st.text_input("Enter your name:")
email = st.text_input("Enter your email:")

# -----------------------
# Save user data
# -----------------------
if name and email:

    if not os.path.exists("users.csv"):
        with open("users.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Email"])

    if not st.session_state.saved:
        with open("users.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, email])
        st.session_state.saved = True

    st.success(f"Welcome {name}! You can now chat.")

    # -----------------------
    # Chat input
    # -----------------------
    user_input = st.text_input("Ask your question:")

    if user_input:
        response = get_response(user_input)

        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("Bot", response))

    # -----------------------
    # Display chat history
    # -----------------------
    for sender, message in st.session_state.history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")

else:
    st.warning("Please enter your name and email first.")