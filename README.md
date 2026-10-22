
# ❤️ CardioClean — ECG Signal Noise Removal

**CardioClean** is a Python-based web application for demonstrating Electrocardiogram (ECG) signal denoising. It generates simulated ECG signals, introduces common types of noise, applies digital filtering, and visualizes the results through an interactive Streamlit dashboard.

The project is developed for educational purposes as part of Digital Signal Processing (DSP) and signal-processing studies.

## 📌 Project Overview

An ECG signal records the electrical activity of the heart. During recording, unwanted disturbances can affect the signal and make its waveform harder to analyze.

CardioClean demonstrates how digital signal processing techniques can help reduce these disturbances while preserving the important features of an ECG waveform.

## ✨ Features

- **ECG Signal Generation:** Generate a simulated ECG waveform.
- **Noise Addition:** Demonstrate power-line interference, baseline wander, and random noise.
- **Signal Filtering:** Apply digital filtering to reduce unwanted noise.
- **Noise Analysis:** Visualize clean and noisy signals for comparison.
- **Performance Evaluation:** Calculate Signal-to-Noise Ratio (SNR) and Root Mean Square Error (RMSE).
- **Interactive Dashboard:** Navigate between signal generation, analysis, filtering, and results.
- **CSV Upload:** Upload supported ECG signal data for analysis.
- **Result Export:** Download processed signal data as a CSV file.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| NumPy | Numerical computations and signal processing |
| Pandas | Data handling and CSV processing |
| Matplotlib | Signal waveform visualization |

## ⚙️ Signal Processing Workflow

1. Generate or upload an ECG signal.
2. Introduce noise to demonstrate signal contamination.
3. Apply a digital filter to the noisy signal.
4. Compare the original, noisy, and filtered waveforms.
5. Evaluate the results using numerical metrics.
6. Export the processed data for further analysis.

## 📂 Project Structure

```text
CardioClean/
├── streamlit_app.py    # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md            # Project documentation
```

The repository may also contain files from the earlier Flask version of the project.

## 🚀 Run the Project Locally

### Prerequisites

- Python 3.10 or a compatible supported Python version
- pip package manager
- Git (optional if you download the repository as a ZIP file)

### Step 1: Download the repository

Clone the repository:

```bash
git clone https://github.com/1022411110-alt/CardioClean.git
cd CardioClean
```

Alternatively, open the repository on GitHub, select **Code → Download ZIP**, and extract the downloaded folder.

### Step 2: Install dependencies

Open a terminal inside the project folder and run:

```bash
python -m pip install -r requirements.txt
```

### Step 3: Launch the application

Run:

```bash
streamlit run streamlit_app.py
```

Streamlit will display a local address in the terminal. Open that address in your web browser to use the application.

## ☁️ Online Deployment

CardioClean can be deployed using Streamlit Community Cloud.

1. Push the project files to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub.
3. Create an app and select the CardioClean repository.
4. Select the `main` branch.
5. Set the main file path to `streamlit_app.py`.
6. Click **Deploy**.

After deployment, Streamlit provides a public web address that can be shared for demonstrations and academic evaluation.

Deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 📊 Evaluation Metrics

**Signal-to-Noise Ratio (SNR):** Measures the signal power relative to noise power. A higher SNR generally indicates a cleaner signal.

**Root Mean Square Error (RMSE):** Measures the difference between a reference signal and a processed signal. A lower RMSE generally indicates a smaller error.

The interpretation of these metrics depends on the reference signal and the noise assumptions used in the experiment.

## 🎓 Applications

- Digital Signal Processing laboratory demonstrations
- ECG signal denoising experiments
- Biomedical signal visualization
- Educational demonstrations of digital filters
- Beginner-level Python and Streamlit projects

## ⚠️ Limitations

- Simulated ECG signals do not fully represent real-world cardiac recordings.
- Filtering performance depends on the type and level of noise.
- Results depend on the quality and format of uploaded data.
- This project is intended for learning and experimentation, not clinical diagnosis or medical decision-making.

## 🔮 Future Improvements

- Support additional ECG datasets and file formats.
- Implement and compare multiple digital filters.
- Add frequency-domain analysis using FFT.
- Display quantitative comparisons before and after filtering.
- Explore adaptive filtering and machine-learning-based denoising.

## 👨‍💻 Author

**Project:** CardioClean — ECG Signal Noise Removal

**Repository:** https://github.com/1022411110-alt/CardioClean

## 📄 License

This project is intended for educational use. A formal open-source license can be added to the repository if the project is to be distributed or reused by others.
