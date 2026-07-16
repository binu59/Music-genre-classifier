import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from src import config
from src.inference import predict_clip

st.set_page_config(page_title="BeatSense", page_icon="🎵", layout="wide", initial_sidebar_state="collapsed")


st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
            background:
                radial-gradient(circle at 20% 10%, rgba(168, 85, 247, 0.18), transparent 26%),
                radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.10), transparent 24%),
                linear-gradient(180deg, #101015 0%, #0b0b10 100%);
            color: #f2efff;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.2rem;
            max-width: 1180px;
        }

        .beatsense-shell {
            background: rgba(13, 13, 20, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.08);
            border-radius: 24px;
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(18px);
            padding: 0.95rem 1.1rem 1.3rem;
        }

        .beatsense-topbar {
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 0.4rem 0.9rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            margin-bottom: 1.35rem;
        }

        .beatsense-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #f4f1ff;
        }

        .beatsense-logo {
            width: 22px;
            height: 22px;
            border-radius: 7px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
            background: linear-gradient(135deg, #b44dff 0%, #8b5cf6 100%);
            box-shadow: 0 0 0 5px rgba(177, 75, 255, 0.08);
        }

        .beatsense-nav {
            display: flex;
            align-items: center;
            gap: 1.1rem;
            color: #a7a3bc;
            font-size: 0.92rem;
        }

        .beatsense-nav span {
            cursor: default;
        }

        .beatsense-nav .active {
            color: #f4f1ff;
        }

        .beatsense-pill,
        .beatsense-hero-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: #b8b2d4;
            font-size: 0.78rem;
            letter-spacing: 0.02em;
        }

        .beatsense-hero {
            text-align: center;
            padding: 0.8rem 0 1.6rem;
        }

        .beatsense-title {
            font-size: clamp(2.3rem, 4vw, 4.15rem);
            line-height: 1.02;
            letter-spacing: -0.06em;
            font-weight: 800;
            margin: 1rem 0 0.9rem;
            color: #f4f1ff;
        }

        .beatsense-title .accent {
            background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .beatsense-subtitle {
            max-width: 640px;
            margin: 0 auto;
            color: #a9a1bf;
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .beatsense-grid {
            display: grid;
            grid-template-columns: 1.55fr 0.95fr;
            gap: 1rem;
            align-items: start;
        }

        .beatsense-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(31, 24, 48, 0.98) 0%, rgba(20, 20, 31, 0.98) 100%);
            border: 1px solid rgba(146, 112, 255, 0.16);
            border-radius: 26px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .beatsense-card--upload {
            min-height: 246px;
            padding: 1.7rem 1.4rem 1.3rem;
            border: 1.5px dashed rgba(181, 84, 255, 0.95);
        }

        .beatsense-upload-glow,
        .beatsense-result-glow {
            position: absolute;
            border-radius: 999px;
            filter: blur(2px);
            pointer-events: none;
        }

        .beatsense-upload-glow {
            width: 136px;
            height: 136px;
            left: -20px;
            top: -18px;
            background: rgba(110, 66, 193, 0.18);
        }

        .beatsense-upload-glow.two {
            width: 120px;
            height: 120px;
            right: -8px;
            bottom: -16px;
            background: rgba(244, 63, 94, 0.12);
        }

        .beatsense-upload-center {
            position: relative;
            z-index: 1;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            gap: 0.75rem;
        }

        .beatsense-icon {
            width: 44px;
            height: 44px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #b44dff 0%, #9858ff 100%);
            color: white;
            font-size: 1.2rem;
            box-shadow: 0 0 0 10px rgba(180, 77, 255, 0.13);
        }

        .beatsense-file-name {
            font-weight: 700;
            color: #f2efff;
            letter-spacing: -0.02em;
        }

        .beatsense-file-meta {
            color: #8f89a9;
            font-size: 0.84rem;
        }

        .beatsense-card--result {
            min-height: 246px;
            padding: 1.3rem 1.2rem 1.2rem;
        }

        .beatsense-card-label {
            color: #8f89a9;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.95rem;
        }

        .beatsense-genre-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 0.35rem 0 1rem;
            padding: 0.55rem 1rem;
            min-width: 190px;
            border-radius: 999px;
            background: linear-gradient(135deg, #ff4c96 0%, #ff5592 100%);
            color: white;
            font-size: clamp(1.35rem, 2vw, 1.95rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            box-shadow: 0 10px 28px rgba(255, 76, 150, 0.28);
        }

        .beatsense-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            color: #b8b2d4;
            font-size: 0.92rem;
        }

        .beatsense-confidence {
            font-size: 1.2rem;
            font-weight: 800;
            color: #f3efff;
        }

        .beatsense-meter {
            margin-top: 0.55rem;
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }

        .beatsense-meter > span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #b44dff 0%, #ff4c96 100%);
        }

        .beatsense-analyzed {
            margin-top: 1rem;
            color: #948bb0;
            font-size: 0.84rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .beatsense-preview {
            margin-top: 1rem;
            padding: 1rem 1rem 1.1rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(28, 25, 46, 0.95) 0%, rgba(21, 19, 35, 0.95) 100%);
            border: 1px solid rgba(148, 112, 255, 0.14);
        }

        .beatsense-preview-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #8d86a7;
            font-size: 0.8rem;
            margin-bottom: 0.75rem;
            font-weight: 600;
        }

        .beatsense-wave {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 58px;
            width: 100%;
        }

        .beatsense-wave span {
            flex: 1 1 0;
            border-radius: 999px;
            background: linear-gradient(180deg, #b44dff 0%, #ff4c96 100%);
            opacity: 0.96;
            min-width: 3px;
        }

        .beatsense-section-title {
            margin: 1.7rem 0 0.75rem;
            color: #f1ecff;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        .beatsense-note {
            color: #8d86a7;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        .stFileUploader {
            position: relative;
            z-index: 1;
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .stFileUploader > div {
            width: 100%;
        }

        .stFileUploader label {
            display: none;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            width: 100%;
            max-width: 760px;
            margin: 0 auto;
            min-height: 104px;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.07);
            background: rgba(255, 255, 255, 0.03);
            padding: 0.9rem 0.95rem;
        }

        .stFileUploader [data-testid="stFileUploaderDropzone"] > div {
            width: 100%;
            color: #d8d2eb;
            text-align: center;
        }

        [data-testid="stHorizontalBlock"] > div {
            min-width: 0;
        }

        .stFileUploader [data-testid="stFileUploaderFile"],
        .stFileUploader [data-testid="stFileUploaderFileNameAndSize"] {
            min-width: 0;
            max-width: 100%;
        }

        .stFileUploader [data-testid="stFileUploaderFile"] {
            display: none;
        }

        .stFileUploader [data-testid="stFileUploaderFileName"] {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .stFileUploader button {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #b44dff 0%, #9f4bff 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 8px 24px rgba(180, 77, 255, 0.25);
        }

        .stButton > button {
            border-radius: 999px;
            border: 0;
            padding: 0.55rem 1rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            background: linear-gradient(135deg, #b44dff 0%, #9c4bff 100%);
            color: white;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #c85cff 0%, #a556ff 100%);
        }

        .stAudio {
            width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return keras.models.load_model(config.MODEL_PATH)


@st.cache_resource
def load_label_names():
    return [str(name) for name in np.load(config.LABEL_NAMES_PATH)]


model = load_model()
label_names = load_label_names()



st.markdown(
    """
    <div class="beatsense-hero">
        <div class="beatsense-title">What genre is that <span class="accent">track?</span></div>
        <div class="beatsense-subtitle">
            Upload any audio clip and our AI model will instantly identify its music genre with confidence scoring.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_spacer, upload_col, right_col = st.columns([1.0, 1.55, 1.0], gap="large")

with upload_col:
    st.markdown(
        """
        <div class="beatsense-card beatsense-card--upload">
            <div class="beatsense-upload-glow"></div>
            <div class="beatsense-upload-glow two"></div>
            <div class="beatsense-upload-center">
                <div class="beatsense-icon">♫</div>
                <div class="beatsense-file-name">Drop your audio file here</div>
                <div class="beatsense-file-meta">WAV files only · best results from clips longer than 3 seconds</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload audio", type=["wav"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown(
            """
            <div class="beatsense-preview" style="margin-top: 1rem;">
                <div class="beatsense-preview-head">
                    <span>Audio Preview</span>
                    <span>Ready to classify</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.audio(uploaded_file, format="audio/wav")

if uploaded_file is not None:
    try:
        with st.spinner("Loading audio..."):
            audio_bytes = uploaded_file.read()
            waveform, sr = librosa.load(io.BytesIO(audio_bytes), sr=config.SAMPLE_RATE, mono=True)

        duration = len(waveform) / config.SAMPLE_RATE
        if duration < config.SEGMENT_DURATION_SEC:
            with right_col:
                st.markdown(
                    f"""
                    <div class="beatsense-card beatsense-card--result">
                        <div class="beatsense-card-label">Predicted genre</div>
                        <div class="beatsense-note">
                            Clip is too short ({duration:.1f}s). Need at least {config.SEGMENT_DURATION_SEC}s.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with st.spinner("Extracting features and predicting..."):
                result = predict_clip(waveform, model, label_names)

            probs_df = pd.DataFrame(
                sorted(result.genre_probabilities.items(), key=lambda x: -x[1]),
                columns=["Genre", "Probability"],
            ).set_index("Genre")

            preview_bars = []
            for chunk in np.array_split(np.abs(waveform), min(24, max(12, len(waveform) // 12000))):
                amplitude = float(np.clip(chunk.mean() * 9.0, 0.12, 1.0))
                preview_bars.append(f'<span style="height:{int(18 + amplitude * 40)}px"></span>')

            with right_col:
                st.markdown(
                    f"""
                    <div class="beatsense-card beatsense-card--result">
                        <div class="beatsense-card-label">Predicted genre</div>
                        <div class="beatsense-genre-pill">{result.predicted_genre.title()}</div>
                        <div class="beatsense-row">
                            <span>Confidence</span>
                            <span class="beatsense-confidence">{result.confidence * 100:.0f}%</span>
                        </div>
                        <div class="beatsense-meter"><span style="width:{result.confidence * 100:.0f}%"></span></div>
                        <div class="beatsense-analyzed">⚡ Analyzed in {result.n_segments_used * config.SEGMENT_DURATION_SEC:.0f}s of audio</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div class='beatsense-section-title'>Genre confidence</div>",
                    unsafe_allow_html=True,
                )
                for genre, probability in probs_df.head(5).itertuples():
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.7rem;">
                            <div class="beatsense-row" style="margin-bottom: 0.35rem; font-size: 0.84rem;">
                                <span>{genre}</span>
                                <span>{probability * 100:.0f}%</span>
                            </div>
                            <div class="beatsense-meter"><span style="width:{probability * 100:.0f}%"></span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Show extracted MFCC (first segment)"):
                from src.feature_extraction import segment_track

                first_segment = segment_track(waveform)[0]
                mfcc = librosa.feature.mfcc(
                    y=first_segment, sr=config.SAMPLE_RATE,
                    n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
                )
                fig, ax = plt.subplots(figsize=(8, 3))
                img = librosa.display.specshow(
                    mfcc, x_axis="time", sr=config.SAMPLE_RATE,
                    hop_length=config.HOP_LENGTH, ax=ax,
                )
                fig.colorbar(img, ax=ax)
                ax.set_title("MFCC — first 3s segment")
                st.pyplot(fig)

    except Exception as e:
        with right_col:
            st.markdown(
                f"""
                <div class="beatsense-card beatsense-card--result">
                    <div class="beatsense-card-label">Prediction error</div>
                    <div class="beatsense-note">Couldn't process this file: {e}</div>
                    <div class="beatsense-note" style="margin-top: 0.85rem;">
                        Make sure it's a valid WAV file. If it was recorded in an unusual format, try re-exporting it as standard 16-bit PCM WAV.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("</div>", unsafe_allow_html=True)
