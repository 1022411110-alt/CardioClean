
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CardioClean | ECG Noise Removal",
    page_icon="💓",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }
    .main-title {
        color: #102b50;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .subtitle {
        color: #617086;
        font-size: 1.05rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e5eaf0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💓 CardioClean</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ECG Signal Noise Removal and Analysis System</div>',
    unsafe_allow_html=True,
)
st.write("")


# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------

FS = 500


def generate_ecg(duration=10, noise_level=0.08, powerline_level=0.12):
    """Generate a simulated ECG, then add artificial noise."""
    t = np.arange(0, duration, 1 / FS)
    clean = np.zeros_like(t)

    # Approximate P, Q, R, S and T waves
    for beat in np.arange(0.5, duration, 1.0):
        clean += 0.12 * np.exp(-((t - (beat - 0.20)) / 0.045) ** 2)
        clean += -0.15 * np.exp(-((t - (beat - 0.045)) / 0.014) ** 2)
        clean += 1.00 * np.exp(-((t - beat) / 0.018) ** 2)
        clean += -0.25 * np.exp(-((t - (beat + 0.035)) / 0.020) ** 2)
        clean += 0.30 * np.exp(-((t - (beat + 0.24)) / 0.085) ** 2)

    rng = np.random.default_rng(42)
    powerline = powerline_level * np.sin(2 * np.pi * 50 * t)
    random_noise = noise_level * rng.normal(size=len(t))
    baseline = 0.08 * np.sin(2 * np.pi * 0.5 * t)

    noisy = clean + powerline + random_noise + baseline
    return t, clean, noisy


# --------------------------------------------------
# FIR LOW-PASS FILTER
# --------------------------------------------------

def apply_filter(signal, cutoff=35.0, order=80):
    """Apply a simple FIR low-pass filter without SciPy."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 2:
        return signal.copy()

    normalized_cutoff = cutoff / (FS / 2)
    taps = np.arange(order + 1) - order / 2

    coefficients = normalized_cutoff * np.sinc(
        normalized_cutoff * taps
    )
    coefficients *= np.hamming(order + 1)
    coefficients /= np.sum(coefficients)

    return np.convolve(signal, coefficients, mode="same")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_snr(reference, observed):
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if len(reference) == 0 or len(reference) != len(observed):
        return 0.0

    noise = observed - reference
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-12:
        return 99.0
    if signal_power < 1e-12:
        return 0.0

    return float(10 * np.log10(signal_power / noise_power))


def calculate_rmse(reference, observed):
    if len(reference) == 0 or len(reference) != len(observed):
        return 0.0
    return float(np.sqrt(np.mean((reference - observed) ** 2)))


def draw_signal(t, values, title, color):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, values, color=color, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Signal Generation",
        "Noise Analysis",
        "Filtering",
        "Results",
        "Upload ECG CSV",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Digital Signal Processing project")
st.sidebar.caption("Educational use only")


# --------------------------------------------------
# LOAD DEFAULT SIGNAL
# --------------------------------------------------

duration = st.sidebar.slider(
    "Signal duration (seconds)", 5, 15, 10
)

noise_level = st.sidebar.slider(
    "Random noise level", 0.0, 0.20, 0.08, 0.01
)

powerline_level = st.sidebar.slider(
    "Power-line noise level", 0.0, 0.25, 0.12, 0.01
)

t, clean, noisy = generate_ecg(
    duration=duration,
    noise_level=noise_level,
    powerline_level=powerline_level,
)

filtered = apply_filter(noisy)


# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "Home":
    st.header("Welcome to CardioClean")

    st.write(
        "CardioClean demonstrates how digital signal processing "
        "can reduce noise in an ECG signal. Explore signal generation, "
        "noise addition, filtering, and quality measurements."
    )

    a, b, c = st.columns(3)
    a.metric("Sampling frequency", f"{FS} Hz")
    b.metric("Signal duration", f"{duration} seconds")
    c.metric("Number of samples", f"{len(t):,}")

    st.subheader("Signal Processing Workflow")

    st.markdown("""
    1. **Generate ECG:** Create a simulated ECG waveform.
    2. **Add noise:** Introduce power-line interference, random noise,
       and baseline wander.
    3. **Apply filter:** Use an FIR low-pass filter.
    4. **Compare results:** Calculate SNR and RMSE.
    """)

    st.info(
        "Use the Navigation menu on the left to explore each project section."
    )

    st.pyplot(draw_signal(t, clean, "Example Original ECG", "#1677c8"))


# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------

elif page == "Signal Generation":
    st.header("1. ECG Signal Generation")
    st.write("The simulated clean ECG waveform is shown below.")

    st.pyplot(draw_signal(t, clean, "Original ECG Signal", "#1677c8"))

    a, b, c = st.columns(3)
    a.metric("Sampling frequency", f"{FS} Hz")
    b.metric("Duration", f"{duration} s")
    c.metric("Samples", f"{len(t):,}")


# --------------------------------------------------
# NOISE ANALYSIS
# --------------------------------------------------

elif page == "Noise Analysis":
    st.header("2. Noise Analysis")
    st.write("Compare the clean signal with the signal containing noise.")

    left, right = st.columns(2)

    with left:
        st.pyplot(draw_signal(t, clean, "Clean ECG", "#1677c8"))

    with right:
        st.pyplot(draw_signal(t, noisy, "Noisy ECG", "#d9534f"))

    st.metric("SNR before filtering", f"{calculate_snr(clean, noisy):.2f} dB")


# --------------------------------------------------
# FILTERING
# --------------------------------------------------

elif page == "Filtering":
    st.header("3. FIR Low-Pass Filtering")

    st.write(
        "The filter uses a 35 Hz cutoff frequency to reduce "
        "high-frequency noise. It will not remove every type of noise."
    )

    st.pyplot(draw_signal(t, noisy, "Noisy ECG", "#d9534f"))
    st.pyplot(draw_signal(t, filtered, "Filtered ECG", "#19966b"))

    st.metric("SNR after filtering", f"{calculate_snr(clean, filtered):.2f} dB")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

elif page == "Results":
    st.header("4. Results and Performance")

    before_snr = calculate_snr(clean, noisy)
    after_snr = calculate_snr(clean, filtered)
    before_rmse = calculate_rmse(clean, noisy)
    after_rmse = calculate_rmse(clean, filtered)

    a, b, c = st.columns(3)
    a.metric("SNR before", f"{before_snr:.2f} dB")
    b.metric("SNR after", f"{after_snr:.2f} dB")
    c.metric("SNR change", f"{after_snr - before_snr:+.2f} dB")

    a, b = st.columns(2)
    a.metric("RMSE before filtering", f"{before_rmse:.4f}")
    b.metric("RMSE after filtering", f"{after_rmse:.4f}")

    st.subheader("Compare all signal stages")

    st.pyplot(draw_signal(t, clean, "Original ECG", "#1677c8"))
    st.pyplot(draw_signal(t, noisy, "Noisy ECG", "#d9534f"))
    st.pyplot(draw_signal(t, filtered, "Filtered ECG", "#19966b"))

    result_df = pd.DataFrame({
        "Time": t,
        "Clean_ECG": clean,
        "Noisy_ECG": noisy,
        "Filtered_ECG": filtered,
    })

    st.download_button(
        "Download ECG Results CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="ECG_Results.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------

elif page == "Upload ECG CSV":
    st.header("Upload Your ECG Data")

    st.write(
        "Upload a CSV with a Clean_ECG column, or a single ECG, "
        "Signal, Amplitude, or Value column. A Time column is optional."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [str(col).strip() for col in df.columns]

            st.subheader("Preview of uploaded data")
            st.dataframe(df.head(10), use_container_width=True)

            lower_columns = {
                col.lower(): col for col in df.columns
            }

            time_col = next(
                (lower_columns[name] for name in
                 ["time", "time_seconds", "timestamp"]
                 if name in lower_columns),
                None,
            )

            clean_col = next(
                (lower_columns[name] for name in
                 ["clean_ecg", "clean", "original_ecg",
                  "ecg", "signal", "amplitude", "value"]
                 if name in lower_columns),
                None,
            )

            if clean_col is None:
                st.error(
                    "Couldn't find an ECG column. Add a column named "
                    "Clean_ECG or ECG and upload again."
                )
            else:
                uploaded_clean = pd.to_numeric(
                    df[clean_col], errors="coerce"
                )

                valid = uploaded_clean.notna()
                uploaded_clean = uploaded_clean[valid].to_numpy()

                if len(uploaded_clean) < 2:
                    st.error("Please upload at least two valid ECG samples.")
                else:
                    if time_col is not None:
                        uploaded_time = pd.to_numeric(
                            df.loc[valid, time_col], errors="coerce"
                        ).to_numpy()
                        if not np.all(np.isfinite(uploaded_time)):
                            uploaded_time = np.arange(
                                len(uploaded_clean)
                            ) / FS
                    else:
                        uploaded_time = np.arange(
                            len(uploaded_clean)
                        ) / FS

                    rng = np.random.default_rng(42)
                    uploaded_noisy = (
                        uploaded_clean
                        + 0.12 * np.sin(
                            2 * np.pi * 50 * uploaded_time
                        )
                        + 0.08 * rng.normal(size=len(uploaded_clean))
                    )
                    uploaded_filtered = apply_filter(uploaded_noisy)

                    st.subheader("Uploaded ECG signal")
                    st.pyplot(draw_signal(
                        uploaded_time,
                        uploaded_clean,
                        "Uploaded ECG",
                        "#1677c8",
                    ))

                    st.pyplot(draw_signal(
                        uploaded_time,
                        uploaded_noisy,
                        "Uploaded ECG with Demonstration Noise",
                        "#d9534f",
                    ))

                    st.pyplot(draw_signal(
                        uploaded_time,
                        uploaded_filtered,
                        "Filtered Uploaded ECG",
                        "#19966b",
                    ))

                    st.caption(
                        "For this upload workflow, demonstration noise is "
                        "added to the uploaded ECG. If the CSV already has "
                        "separate noisy and filtered columns, this example "
                        "does not use those columns."
                    )

                    uploaded_results = pd.DataFrame({
                        "Time": uploaded_time,
                        "Clean_ECG": uploaded_clean,
                        "Noisy_ECG": uploaded_noisy,
                        "Filtered_ECG": uploaded_filtered,
                    })

                    st.download_button(
                        "Download Uploaded ECG Results",
                        data=uploaded_results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="Uploaded_ECG_Results.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error(f"Couldn't process this CSV: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "CardioClean is an educational signal-processing demonstration, "
    "not a medical diagnostic tool."
)
