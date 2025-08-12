# app.py
"""
Streamlit English Practice Tutor — Advanced version
Features:
- Web GUI (mobile + desktop)
- Upload audio or use browser recording (if you add streamlit-webrtc)
- Transcription via OpenAI Whisper (whisper-1)
- Chat replies and corrections via OpenAI Chat
- Proxy pronunciation scoring (word-accuracy + Levenshtein)
- TTS playback using gTTS (simple, offline-like)
- Session logging (CSV) and dashboard
Notes:
- Set your OPENAI_API_KEY in the app UI or via env var / secrets on deployment.
- For production-grade phoneme-level scoring, integrate Azure Pronunciation Assessment API (optional).
"""
import streamlit as st
import openai
import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import base64
import Levenshtein
import uuid
import pathlib

# ---------------- config ----------------
LOG_CSV = "practice_logs.csv"
st.set_page_config(page_title="English Practice Tutor", layout="wide")

# -------------- helpers ------------------
@st.cache_resource
def init_openai(api_key: str):
    openai.api_key = api_key
    return True

def save_audio_bytes_to_file(audio_bytes, suffix=".wav"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name

def transcribe_with_openai_whisper(audio_path, model="whisper-1"):
    """
    Use OpenAI Audio.transcribe(...). Returns text or empty string.
    """
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = os.path.basename(audio_path)
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
        # transcription is a dict-like with 'text'
        text = transcription.get("text") if isinstance(transcription, dict) else getattr(transcription, "text", "")
        return text or ""
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return ""

def chat_reply(messages, model="gpt-4o"):
    try:
        resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=500, temperature=0.7)
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"OpenAI chat error: {e}")
        return "Sorry, I couldn't reach the chat model."

def tts_save_mp3(text):
    """
    Save speech to an MP3 file via gTTS and return file path.
    """
    tts = gTTS(text=text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def compute_pronunciation_score(target_sentence, asr_text):
    """
    Lightweight proxy scoring:
      - Word accuracy: fraction of target words found in ASR
      - Levenshtein similarity over joined strings
    """
    def normalize(s):
        return ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in (s or "")).split()

    target_words = normalize(target_sentence)
    asr_words = normalize(asr_text)
    if not target_words:
        return {"word_accuracy": 0.0, "levenshtein_score": 0.0, "matched_words": 0, "target_word_count": 0}
    matched = sum(1 for w in target_words if w in asr_words)
    word_accuracy = matched / len(target_words) * 100.0
    targ_join = ' '.join(target_words)
    asr_join = ' '.join(asr_words)
    max_len = max(len(targ_join), 1)
    lev_dist = Levenshtein.distance(targ_join, asr_join)
    lev_score = max(0.0, (1 - lev_dist / max_len)) * 100.0
    return {"word_accuracy": round(word_accuracy,1), "levenshtein_score": round(lev_score,1), "matched_words": matched, "target_word_count": len(target_words)}

def append_log(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)

def load_logs():
    if os.path.exists(LOG_CSV):
        return pd.read_csv(LOG_CSV)
    else:
        return pd.DataFrame(columns=["timestamp","session_id","prompt","transcript","word_accuracy","lev_score","bot_reply"])

# --------------- UI ---------------------
st.title("English Practice Tutor — Advanced")
st.markdown("Practice speaking, get corrections, track progress, and improve pronunciation.")

with st.sidebar:
    st.header("Settings & API Key")
    OPENAI_API_KEY = st.text_input("OpenAI API Key (paste here or set as secret)", type="password")
    if OPENAI_API_KEY:
        init_openai(OPENAI_API_KEY)
    model_choice = st.selectbox("Chat model", options=["gpt-4o","gpt-4o-mini","gpt-4o-mini-transcribe"], index=0)
    enable_tts = st.checkbox("Enable TTS (gTTS)", value=True)
    enable_file_download = st.checkbox("Allow log download", value=True)
    st.markdown("Tip: For better pronunciation scoring integrate Azure Pronunciation Assessment (optional).")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Record or Upload Audio")
    st.markdown("Record on your phone (Voice Recorder app) and upload the file here, or use a browser recorder plugin/streamlit-webrtc.")
    audio_file = st.file_uploader("Upload audio (wav/mp3/m4a/webm)", type=["wav","mp3","m4a","webm"])
    target_sentence = st.text_input("Target sentence (optional) — type sentence you'd like to practise", value="")
    start_practice = st.button("Transcribe & Evaluate")

with col2:
    st.subheader("Type / Chat")
    user_text = st.text_area("Type any sentence or ask the tutor for exercises/corrections.")
    send_chat = st.button("Send to Tutor")

# Process audio upload
if start_practice and OPENAI_API_KEY:
    if audio_file is None:
        st.warning("Please upload an audio file recorded on your phone.")
    else:
        audio_bytes = audio_file.read()
        audio_path = save_audio_bytes_to_file(audio_bytes, suffix=os.path.splitext(audio_file.name)[1])
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_with_openai_whisper(audio_path)
        st.markdown("**Transcript:**")
        st.write(transcript or "— (empty) —")

        if target_sentence.strip():
            score = compute_pronunciation_score(target_sentence, transcript)
            st.markdown("**Pronunciation (proxy)**")
            st.metric("Word accuracy (%)", score['word_accuracy'])
            st.metric("Levenshtein similarity (%)", score['levenshtein_score'])
        else:
            score = {"word_accuracy": None, "levenshtein_score": None}

        # Chat feedback from model
        messages = [
            {"role":"system","content":"You are a friendly English tutor. Keep replies short and actionable. Provide grammar correction and pronunciation tips when asked."},
            {"role":"user","content": f"I spoke this: \"{transcript}\". Give concise feedback (1-3 sentences) on grammar and give 1 pronunciation tip."}
        ]
        with st.spinner("Getting tutor feedback..."):
            bot_reply = chat_reply(messages, model=model_choice)
        st.markdown("**Tutor feedback:**")
        st.write(bot_reply)

        # TTS playback
        if enable_tts and bot_reply:
            try:
                mp3_path = tts_save_mp3(bot_reply)
                st.audio(open(mp3_path, "rb").read(), format='audio/mp3')
            except Exception as e:
                st.warning(f"TTS failed: {e}")

        # Save log
        session_row = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "prompt": target_sentence,
            "transcript": transcript,
            "word_accuracy": score.get('word_accuracy'),
            "lev_score": score.get('levenshtein_score'),
            "bot_reply": bot_reply
        }
        append_log(session_row)
        st.success("Saved session to logs.")

# Process typed chat
if send_chat and OPENAI_API_KEY and user_text.strip():
    messages = [
        {"role":"system","content":"You are an English tutor. Provide short corrections and practice prompts."},
        {"role":"user","content": user_text}
    ]
    with st.spinner("Getting reply..."):
        bot_reply = chat_reply(messages, model=model_choice)
    st.markdown("**Tutor reply:**")
    st.write(bot_reply)
    if enable_tts:
        try:
            mp3_path = tts_save_mp3(bot_reply)
            st.audio(open(mp3_path, "rb").read(), format='audio/mp3')
        except Exception as e:
            st.warning(f"TTS failed: {e}")

    # Save typed chat
    session_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": str(uuid.uuid4()),
        "prompt": user_text,
        "transcript": user_text,
        "word_accuracy": None,
        "lev_score": None,
        "bot_reply": bot_reply
    }
    append_log(session_row)
    st.success("Saved chat to logs.")

# Dashboard
st.markdown("---")
st.header("Progress Dashboard")
logs = load_logs()
if logs.empty:
    st.info("No practice logs yet.")
else:
    st.write(f"Total sessions: {len(logs)}")
    logs['timestamp'] = pd.to_datetime(logs['timestamp'])
    try:
        st.line_chart(logs[['timestamp','word_accuracy']].set_index('timestamp').fillna(method='ffill'))
    except Exception:
        pass
    st.dataframe(logs.sort_values('timestamp', ascending=False).reset_index(drop=True))
    if enable_file_download:
        csv_bytes = logs.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="practice_logs.csv">Download practice logs</a>'
        st.markdown(href, unsafe_allow_html=True)

st.markdown("Tip: For more accurate pronunciation scoring we can add Azure Pronunciation Assessment. Ask me to add it and I'll provide the code & setup steps.")
