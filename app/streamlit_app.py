import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
if "page_config_done" not in st.session_state:
    st.set_page_config(page_title="Krishna Ji: Devotional Intelligence Dashboard", layout="centered")
    st.session_state["page_config_done"] = True
from streamlit_mic_recorder import mic_recorder
from src.pipeline import InferencePipeline


st.title("Krishna Ji: Devotional Intelligence Dashboard")
st.caption("Speak or type your message. The dashboard will analyze sentiment, toxicity, and devotional topic.")

@st.cache_resource
def get_pipeline():
    return InferencePipeline()

pipeline = get_pipeline()

with st.expander("Settings"):
    colr1, colr2 = st.columns(2)
    with colr1:
        reload_btn = st.button("Reload models")
    if reload_btn:
        try:
            st.cache_resource.clear()
            pipeline = get_pipeline()
            st.success("Models reloaded.")
        except Exception as e:
            st.warning(f"Reload failed: {e}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Microphone")
    audio_data = mic_recorder(key="mic", start_prompt="Start recording", stop_prompt="Stop recording", format="wav")
with col2:
    st.subheader("Upload Audio")
    uploaded_audio = st.file_uploader("Upload a WAV/MP3 audio file", type=["wav", "mp3"])

auto_transcribe = st.checkbox("Auto-transcribe recorded audio", value=True)
instant_classify = st.checkbox("Instant classify on record stop", value=True)
if auto_transcribe and audio_data and "bytes" in audio_data and audio_data["bytes"]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        tf.write(audio_data["bytes"])
        temp_path = tf.name
    st.session_state["live_transcription"] = pipeline.transcribe(temp_path)
    if instant_classify and st.session_state.get("live_transcription"):
        preds = pipeline.analyze_text(st.session_state["live_transcription"])
        st.markdown("### Live Predictions")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Sentiment", value=preds["sentiment"]["label"])
            st.progress(min(max(preds["sentiment"]["confidence"], 0.0), 1.0))
        with c2:
            st.metric(label="Toxicity", value=preds["toxicity"]["label"])
            st.progress(min(max(preds["toxicity"]["confidence"], 0.0), 1.0))
        with c3:
            st.metric(label="Topic", value=preds["topic"]["label"])
            st.progress(min(max(preds["topic"]["confidence"], 0.0), 1.0))
        import pandas as pd
        df = pd.DataFrame(
            [
                {"Category": "Sentiment", "Result": preds["sentiment"]["label"], "Confidence": f"{preds['sentiment']['confidence']:.2f}"},
                {"Category": "Toxicity", "Result": preds["toxicity"]["label"], "Confidence": f"{preds['toxicity']['confidence']:.2f}"},
                {"Category": "Topic", "Result": preds["topic"]["label"], "Confidence": f"{preds['topic']['confidence']:.2f}"},
            ]
        )
        st.table(df)

st.subheader("Or Type Text")
typed_text = st.text_area("Type here", height=120, placeholder="Share your thoughts, e.g., 'I feel anxious about my job interview'", key="typed_text")
st.subheader("Live Transcription")
st.text_area("Transcribed Live", st.session_state.get("live_transcription", ""), height=120, key="live_transcription_box")

analyze_clicked = st.button("Analyze")

transcription = ""
if analyze_clicked:
    if typed_text and typed_text.strip():
        transcription = typed_text.strip()
    else:
        temp_audio_path = None
        if audio_data and "bytes" in audio_data and audio_data["bytes"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                tf.write(audio_data["bytes"])
                temp_audio_path = tf.name
        elif uploaded_audio is not None:
            suffix = Path(uploaded_audio.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(uploaded_audio.read())
                temp_audio_path = tf.name
        if not transcription and st.session_state.get("live_transcription"):
            transcription = st.session_state.get("live_transcription")
        if not transcription and temp_audio_path:
            transcription = pipeline.transcribe(temp_audio_path)
        else:
            st.warning("Please record audio, upload a file, or type text.")
    if transcription:
        preds = pipeline.analyze_text(transcription)
        st.markdown("### Transcription")
        st.text_area("Transcribed Text", transcription, height=120, key="final_transcription")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Sentiment")
            st.metric(label="Label", value=preds["sentiment"]["label"])
            st.progress(min(max(preds["sentiment"]["confidence"], 0.0), 1.0))
        with c2:
            st.markdown("### Toxicity")
            st.metric(label="Label", value=preds["toxicity"]["label"])
            st.progress(min(max(preds["toxicity"]["confidence"], 0.0), 1.0))
        with c3:
            st.markdown("### Topic")
            st.metric(label="Label", value=preds["topic"]["label"])
            st.progress(min(max(preds["topic"]["confidence"], 0.0), 1.0))
        st.markdown("### Predictions")
        df = pd.DataFrame(
            [
                {"Category": "Sentiment", "Result": preds["sentiment"]["label"], "Confidence": f"{preds['sentiment']['confidence']:.2f}"},
                {"Category": "Toxicity", "Result": preds["toxicity"]["label"], "Confidence": f"{preds['toxicity']['confidence']:.2f}"},
                {"Category": "Topic", "Result": preds["topic"]["label"], "Confidence": f"{preds['topic']['confidence']:.2f}"},
            ]
        )
        st.table(df)
