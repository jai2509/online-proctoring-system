import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import face_recognition
import numpy as np
import time
import random
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Proctoring System", layout="wide")
st.title("🧠 AI-Based Proctoring System")

# ==============================
# Session State Initialization
# ==============================
if 'log' not in st.session_state:
    st.session_state.log = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz' not in st.session_state:
    st.session_state.quiz = []

# ==============================
# Helper Functions
# ==============================
def log_event(event):
    timestamp = time.time() - st.session_state.start_time
    st.session_state.log.append((round(timestamp, 2), event))

def generate_quiz(num_questions, duration):
    prompt = f"Generate a {num_questions}-question multiple choice quiz with answers on general knowledge. Each question must have 4 options and correct answer marked."
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error generating quiz."

# ==============================
# Video Processor
# ==============================
class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_face_count = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_count = len(face_locations)

        if face_count != self.prev_face_count:
            if face_count == 0:
                log_event("No face detected")
            elif face_count == 1:
                log_event("One face detected")
            else:
                log_event(f"Multiple faces detected: {face_count}")
            self.prev_face_count = face_count

        return image

# ==============================
# Streamlit UI
# ==============================
st.header("📷 Webcam Monitoring")
st.markdown("Proctoring in progress... Please stay visible.")
webrtc_streamer(key="example", video_processor_factory=FaceDetectionTransformer)

# ==============================
# Quiz Section
# ==============================
st.subheader("📝 Quiz Configuration")
num_q = st.number_input("Enter number of questions:", min_value=1, max_value=25, value=5)
duration = st.number_input("Duration in minutes:", min_value=1, max_value=60, value=5)

if st.button("Generate Quiz"):
    st.session_state.quiz = generate_quiz(num_q, duration)
    st.session_state.quiz_started = True
    st.session_state.start_time = time.time()
    log_event("Quiz started")

if st.session_state.quiz_started:
    st.subheader("🧠 Quiz")
    st.markdown(st.session_state.quiz)
    if st.button("Submit Quiz"):
        st.session_state.quiz_submitted = True
        log_event("Quiz submitted")

# ==============================
# Log Summary & Export
# ==============================
if st.session_state.quiz_submitted:
    st.subheader("📊 Proctoring Log")
    for entry in st.session_state.log:
        st.write(f"Time: {entry[0]}s - Event: {entry[1]}")

    if st.download_button("Download Log as CSV", data="\n".join([f"{t},{e}" for t, e in st.session_state.log]), file_name="proctoring_log.csv"):
        st.success("Log downloaded successfully.")
