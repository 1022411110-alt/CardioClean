
from flask import Flask, request, render_template_string, Response
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
import csv

app = Flask(__name__)

# =========================================================
# CARDIOCLEAN - ECG SIGNAL NOISE REMOVAL WEB APPLICATION
# =========================================================

FS = 500
DURATION = 10

BG = "#0b1220"
BLUE = "#55b8ff"
GREEN = "#42dfa0"
RED = "#ff7185"
YELLOW = "#ffd166"

time_data = np.array([])
clean_data = np.array([])
noisy_data = np.array([])
filtered_data = np.array([])
data_source = "Built-in simulated ECG"


# ---------------- SIGNAL GENERATION ----------------

def generate_signal():
    global time_data, clean_data, noisy_data, filtered_data

    time_data = np.arange(0, DURATION, 1 / FS)
    clean_data = np.zeros_like(time_data)

    # Generate an educational ECG-like waveform.
    for beat in range(10):
        center = beat + 0.5

        clean_data += 0.12 * np.exp(
            -((time_data - (center - 0.20)) / 0.04) ** 2
        )
        clean_data -= 0.15 * np.exp(
            -((time_data - (center - 0.045)) / 0.015) ** 2
        )
        clean_data += 1.00 * np.exp(
            -((time_data - center) / 0.018) ** 2
        )
        clean_data -= 0.25 * np.exp(
            -((time_data - (center + 0.04)) / 0.02) ** 2
        )
        clean_data += 0.30 * np.exp(
            -((time_data - (center + 0.25)) / 0.07) ** 2
        )

    rng = np.random.default_rng(7)
    powerline = 0.25 * np.sin(2 * np.pi * 50 * time_data)
    random_noise = 0.15 * rng.normal(size=len(time_data))

    noisy_data = clean_data + powerline + random_noise
    filtered_data = apply_filter(noisy_data)


def apply_filter(signal):
    """Apply an FIR low-pass filter without extra toolboxes."""
    order = 80
    cutoff = 35
    n = np.arange(order + 1)
    normalized_cutoff = cutoff / (FS / 2)

    coefficients = normalized_cutoff * np.sinc(
        normalized_cutoff * (n - order / 2)
    )
    coefficients *= np.hamming(order + 1)
    coefficients /= np.sum(coefficients)

    return np.convolve(signal, coefficients, mode="same")


def calculate_snr(reference, observed):
    noise = observed - reference
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-12:
        return 99.0

    return 10 * np.log10(
        max(signal_power, 1e-12) / noise_power
    )


# ---------------- GRAPHS ----------------

def graph_image(lines, title, xlabel="Time (seconds)",
                ylabel="Amplitude"):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor("#151f32")
    ax.set_facecolor("#151f32")

    for x, y, label, color in lines:
        ax.plot(x, y, label=label, color=color, linewidth=1)

    ax.set_title(title, color="white", loc="left", pad=12)
    ax.set_xlabel(xlabel, color="#a9b8ce")
    ax.set_ylabel(ylabel, color="#a9b8ce")
    ax.tick_params(colors="#a9b8ce")
    ax.grid(True, alpha=0.2)

    for spine in ax.spines.values():
        spine.set_color("#40516b")

    legend = ax.legend(facecolor="#1d2a40", edgecolor="#1d2a40")
    for item in legend.get_texts():
        item.set_color("white")

    fig.tight_layout()

    output = BytesIO()
    fig.savefig(
        output, format="png", dpi=110,
        facecolor=fig.get_facecolor()
    )
    plt.close(fig)

    return base64.b64encode(output.getvalue()).decode("ascii")


def graph_html(encoded, alt):
    return (
        '<div class="card">'
        f'<img class="graph" src="data:image/png;base64,{encoded}" '
        f'alt="{alt}">'
        '</div>'
    )


def metric(title, value, color):
    return (
        '<div class="metric">'
        f'<div class="label">{title}</div>'
        f'<div class="value {color}">{value}</div>'
        '</div>'
    )


def metrics_grid(items):
    return '<div class="grid">' + "".join(
        metric(a, b, c) for a, b, c in items
    ) + '</div>'


# ---------------- WEBSITE TEMPLATE ----------------

PAGES = [
    ("Home", "home"),
    ("Signal Generation", "signal"),
    ("Noise Addition", "noise"),
    ("Noise Removal", "filter_page"),
    ("Results", "results")
]

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }} | CardioClean</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #0b1220;
    color: #f3f6fc;
    font-family: Arial, Helvetica, sans-serif;
}
header {
    background: #101a2b;
    padding: 20px 5%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #26354b;
    gap: 15px;
}
.logo { color: #55b8ff; font-size: 25px; font-weight: bold; }
.small { color: #a9b8ce; font-size: 12px; }
nav {
    display: flex;
    gap: 8px;
    padding: 14px 5%;
    background: #111b2d;
    flex-wrap: wrap;
}
nav a {
    color: #d7e2f2;
    text-decoration: none;
    padding: 10px 13px;
    border-radius: 7px;
    font-size: 14px;
}
nav a:hover, nav a.active {
    background: #243750;
    color: #55b8ff;
}
main { max-width: 1200px; margin: auto; padding: 28px 24px; }
h1 { font-size: 30px; margin: 0 0 10px; }
h2 { font-size: 20px; }
p { line-height: 1.7; color: #d4dfef; }
.subtitle { color: #a9b8ce; margin-bottom: 25px; }
.hero {
    background: linear-gradient(130deg, #172c49, #15243a);
    border: 1px solid #2c4260;
    border-radius: 18px;
    padding: 34px;
    margin: 20px 0;
}
.hero h1 { color: #55b8ff; font-size: 35px; }
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 15px;
    margin: 22px 0;
}
.card {
    background: #151f32;
    border: 1px solid #26354b;
    border-radius: 13px;
    padding: 21px;
    margin: 15px 0;
}
.metric {
    background: #1d2a40;
    border-radius: 12px;
    padding: 19px;
}
.metric .label { color: #a9b8ce; font-size: 13px; }
.metric .value {
    font-size: 25px;
    font-weight: bold;
    margin-top: 10px;
}
.blue { color: #55b8ff; }
.green { color: #42dfa0; }
.red { color: #ff7185; }
.yellow { color: #ffd166; }
.graph {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 10px;
    margin-top: 10px;
}
.button {
    display: inline-block;
    background: #42dfa0;
    color: #0b1220;
    text-decoration: none;
    font-weight: bold;
    padding: 13px 20px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    margin: 6px 5px 6px 0;
}
.button.bluebtn { background: #55b8ff; }
.button.secondary { background: #243750; color: white; }
.button.yellowbtn { background: #ffd166; }
.actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 25px;
    gap: 10px;
}
.upload-message {
    padding: 12px;
    border-radius: 8px;
    overflow-wrap: anywhere;
}
.upload-message:empty { display: none; }
.upload-message.success { color: #42dfa0; background: #12352f; }
.upload-message.error { color: #ff7185; background: #3a202b; }
.upload-message.info { color: #55b8ff; background: #172c49; }
input[type=file] {
    display: block;
    color: #d4dfef;
    margin: 15px 0;
    max-width: 100%;
}
footer {
    text-align: center;
    color: #8799b3;
    border-top: 1px solid #26354b;
    padding: 20px;
    font-size: 12px;
}
@media(max-width:650px) {
    header { align-items: flex-start; flex-direction: column; }
    h1 { font-size: 25px; }
    .hero { padding: 22px; }
    main { padding: 22px 15px; }
}
</style>
</head>
<body>
<header>
    <div class="logo">♥ CARDIOCLEAN</div>
    <div class="small">ECG SIGNAL NOISE REMOVAL SYSTEM</div>
</header>
<nav>
    {% for item in pages %}
    <a href="{{ url_for(item[1]) }}"
       class="{% if endpoint == item[1] %}active{% endif %}">
       {{ item[0] }}
    </a>
    {% endfor %}
</nav>
<main>
    {{ body|safe }}
    <div class="actions">
        {% if previous %}
        <a class="button secondary" href="{{ url_for(previous) }}">
            ← Previous
        </a>
        {% else %}
        <span></span>
        {% endif %}
        {% if next_page %}
        <a class="button" href="{{ url_for(next_page) }}">
            {% if endpoint == 'home' %}START PROJECT →{% else %}NEXT →{% endif %}
        </a>
        {% endif %}
    </div>
</main>
<footer>
    CARDIOCLEAN · DIGITAL SIGNAL PROCESSING · EDUCATIONAL SIMULATION
</footer>
</body>
</html>
"""


def render_page(title, body, endpoint, previous=None, next_page=None):
    return render_template_string(
        BASE_HTML,
        title=title,
        body=body,
        endpoint=endpoint,
        pages=PAGES,
        previous=previous,
        next_page=next_page
    )


# ---------------- HOME ----------------

@app.route("/")
def home():
    body = """
    <div class="hero">
        <p class="blue"><b>WELCOME TO YOUR DSP PROJECT</b></p>
        <h1>CLEAN SIGNALS.<br>CLEARER ANALYSIS.</h1>
        <p>
            Explore how digital filtering can reduce unwanted noise
            from an ECG signal. Follow the complete process from
            signal generation to the final results.
        </p>
        <a class="button" href="/signal">START PROJECT →</a>
    </div>
    """ + metrics_grid([
        ("SAMPLING RATE", "500 Hz", "blue"),
        ("DURATION", "10 seconds", "green"),
        ("FILTER", "FIR Low-pass", "yellow")
    ]) + """
    <div class="card">
        <h2>Project Objective</h2>
        <p>
            Generate or import ECG data, inspect noise, apply a digital
            filter, and compare signal quality using graphs and SNR.
        </p>
    </div>
    <div class="card">
        <h2>Technologies</h2>
        <p>Python, Flask, NumPy, Matplotlib and MATLAB Online.</p>
    </div>
    """
    return render_page("Home", body, "home", next_page="signal")


# ---------------- SIGNAL GENERATION + CSV UPLOAD ----------------

@app.route("/signal", methods=["GET", "POST"])
def signal():
    global time_data, clean_data, noisy_data, filtered_data, data_source

    message = ""
    message_type = "info"

    if request.method == "POST":
        uploaded = request.files.get("ecg_file")

        if not uploaded or not uploaded.filename:
            message = "Please choose a CSV file first."
            message_type = "error"
        elif not uploaded.filename.lower().endswith(".csv"):
            message = "Please upload a .csv file, not an image."
            message_type = "error"
        else:
            try:
                content = uploaded.read().decode("utf-8-sig")
                rows = list(csv.DictReader(StringIO(content)))

                required = [
                    "Time", "Clean_ECG", "Noisy_ECG", "Filtered_ECG"
                ]

                if not rows:
                    raise ValueError("The selected CSV is empty.")

                if any(column not in rows[0] for column in required):
                    raise ValueError(
                        "Required columns: Time, Clean_ECG, Noisy_ECG, "
                        "Filtered_ECG. Please use ECG_Results.csv from "
                        "our MATLAB script."
                    )

                new_time = np.array(
                    [float(row["Time"]) for row in rows]
                )
                new_clean = np.array(
                    [float(row["Clean_ECG"]) for row in rows]
                )
                new_noisy = np.array(
                    [float(row["Noisy_ECG"]) for row in rows]
                )
                new_filtered = np.array(
                    [float(row["Filtered_ECG"]) for row in rows]
                )

                if len(new_time) < 2:
                    raise ValueError("The CSV needs at least two samples.")

                if not all(
                    len(a) == len(new_time)
                    for a in [new_clean, new_noisy, new_filtered]
                ):
                    raise ValueError("The CSV columns have different lengths.")

                if not all(
                    np.all(np.isfinite(a))
                    for a in [new_time, new_clean, new_noisy, new_filtered]
                ):
                    raise ValueError("The CSV contains invalid numbers.")

                if np.any(np.diff(new_time) <= 0):
                    raise ValueError(
                        "Time values must be strictly increasing."
                    )

                # Only replace current data after successful validation.
                time_data = new_time
                clean_data = new_clean
                noisy_data = new_noisy
                filtered_data = new_filtered
                data_source = uploaded.filename

                message = (
                    f"Loaded successfully! {len(new_time):,} samples "
                    f"imported from {uploaded.filename}."
                )
                message_type = "success"

            except Exception as error:
                message = f"Could not load CSV: {error}"
                message_type = "error"

    duration = time_data[-1] - time_data[0]
    encoded = graph_image(
        [(time_data, clean_data, "Original ECG", BLUE)],
        "Original ECG Signal"
    )

    body = """
    <h1>01 | Signal Generation</h1>
    <p class="subtitle">
        Upload your MATLAB ECG CSV, or use the built-in simulated signal.
    </p>

    <div class="card">
        <h2>Upload Your ECG CSV</h2>
        <p>
            Select ECG_Results.csv generated by ECG_Noise_Removal.m.
            The imported data will be used throughout the dashboard.
        </p>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="ecg_file" accept=".csv" required>
            <button class="button" type="submit">UPLOAD ECG DATA</button>
        </form>
        <p class="upload-message {{ message_type }}">{{ message }}</p>
    </div>
    """ + metrics_grid([
        ("DATA SOURCE", data_source, "blue"),
        ("DURATION", f"{duration:.2f} seconds", "green"),
        ("TOTAL SAMPLES", f"{len(time_data):,}", "yellow")
    ]) + graph_html(encoded, "Original ECG signal") + """
    <div class="card">
        <h2>Observation</h2>
        <p>
            The graph shows the original signal currently loaded.
            Continue to Noise Addition to inspect the effect of noise.
        </p>
    </div>
    """

    body = render_template_string(
        body, message=message, message_type=message_type
    )

    return render_page(
        "Signal Generation", body, "signal",
        previous="home", next_page="noise"
    )


# ---------------- NOISE ADDITION ----------------

@app.route("/noise")
def noise():
    before = calculate_snr(clean_data, noisy_data)

    encoded = graph_image([
        (time_data, noisy_data, "Noisy ECG", RED)
    ], "Noisy ECG Signal")

    body = """
    <h1>02 | Noise Addition</h1>
    <p class="subtitle">Inspect the ECG signal after noise has been added.</p>
    <div class="card">
        <h2>What is noise?</h2>
        <p>
            Noise is an unwanted change in a signal. In this demonstration,
            the supplied data contains a noisy ECG column. For the built-in
            data, 50 Hz interference and random noise are added.
        </p>
    </div>
    """ + metrics_grid([
        ("SNR BEFORE FILTERING", f"{before:.2f} dB", "red"),
        ("DATA SOURCE", data_source, "blue")
    ]) + graph_html(encoded, "Noisy ECG signal") + """
    <div class="card">
        <h2>Observation</h2>
        <p>Notice the unwanted variations in the noisy waveform.</p>
    </div>
    """
    return render_page(
        "Noise Addition", body, "noise",
        previous="signal", next_page="filter_page"
    )


# ---------------- NOISE REMOVAL ----------------

@app.route("/filter")
def filter_page():
    before_graph = graph_image([
        (time_data, noisy_data, "Noisy ECG", RED)
    ], "Noisy ECG before filtering")

    after_graph = graph_image([
        (time_data, filtered_data, "Filtered ECG", GREEN)
    ], "Filtered ECG after noise removal")

    frequencies = np.fft.rfftfreq(
        len(time_data),
        d=(time_data[1] - time_data[0])
    )
    noisy_spectrum = np.abs(np.fft.rfft(noisy_data)) / len(time_data)
    filtered_spectrum = (
        np.abs(np.fft.rfft(filtered_data)) / len(time_data)
    )

    spectrum = graph_image([
        (frequencies, noisy_spectrum, "Noisy", RED),
        (frequencies, filtered_spectrum, "Filtered", GREEN)
    ], "Frequency Spectrum", "Frequency (Hz)", "Magnitude")

    body = """
    <h1>03 | Noise Removal</h1>
    <p class="subtitle">Compare the signal before and after filtering.</p>
    <div class="card">
        <h2>FIR Low-pass Filter</h2>
        <p>
            The filter reduces higher-frequency components. It may also
            remove useful signal details, so results must be interpreted
            carefully.
        </p>
    </div>
    <h2 class="red">Before Filtering</h2>
    """ + graph_html(before_graph, "Noisy signal") + """
    <h2 class="green">After Filtering</h2>
    """ + graph_html(after_graph, "Filtered signal") + """
    <h2 class="blue">Frequency Spectrum Comparison</h2>
    """ + graph_html(spectrum, "Frequency spectrum") + """
    <div class="card">
        <p>
            The built-in signal uses an FIR low-pass filter with a 35 Hz
            cutoff. Imported Filtered_ECG data is displayed as provided
            by the uploaded CSV.
        </p>
    </div>
    """
    return render_page(
        "Noise Removal", body, "filter_page",
        previous="noise", next_page="results"
    )


# ---------------- RESULTS + CSV UPLOAD ----------------

@app.route("/results", methods=["GET", "POST"])
def results():
    global time_data, clean_data, noisy_data, filtered_data, data_source

    message = ""

    # Optional CSV upload on the Results page as well.
    if request.method == "POST":
        uploaded = request.files.get("csv_file")

        if uploaded and uploaded.filename:
            try:
                rows = list(csv.DictReader(
                    StringIO(uploaded.read().decode("utf-8-sig"))
                ))
                required = [
                    "Time", "Clean_ECG", "Noisy_ECG", "Filtered_ECG"
                ]

                if not rows or any(k not in rows[0] for k in required):
                    raise ValueError("CSV columns do not match the expected format.")

                nt = np.array([float(r["Time"]) for r in rows])
                nc = np.array([float(r["Clean_ECG"]) for r in rows])
                nn = np.array([float(r["Noisy_ECG"]) for r in rows])
                nf = np.array([float(r["Filtered_ECG"]) for r in rows])

                if len(nt) < 2 or not all(
                    len(a) == len(nt) for a in [nc, nn, nf]
                ):
                    raise ValueError("Invalid or incomplete CSV columns.")

                if not all(np.all(np.isfinite(a)) for a in [nt, nc, nn, nf]):
                    raise ValueError("CSV contains invalid numeric values.")

                if np.any(np.diff(nt) <= 0):
                    raise ValueError("Time values must be increasing.")

                time_data, clean_data = nt, nc
                noisy_data, filtered_data = nn, nf
                data_source = uploaded.filename
                message = "CSV loaded successfully."

            except Exception as error:
                message = f"Could not load CSV: {error}"
        else:
            message = "Please choose a CSV file."

    before = calculate_snr(clean_data, noisy_data)
    after = calculate_snr(clean_data, filtered_data)
    change = after - before

    original_graph = graph_image([
        (time_data, clean_data, "Original ECG", BLUE)
    ], "Original ECG Signal")

    noisy_graph = graph_image([
        (time_data, noisy_data, "Noisy ECG", RED)
    ], "Noisy ECG Signal")

    filtered_graph = graph_image([
        (time_data, filtered_data, "Filtered ECG", GREEN)
    ], "Filtered ECG Signal")

    body = """
    <h1>04 | Results & Conclusion</h1>
    <p class="subtitle">Review the three signals separately.</p>
    """ + metrics_grid([
        ("SNR BEFORE", f"{before:.2f} dB", "red"),
        ("SNR AFTER", f"{after:.2f} dB", "green"),
        ("SNR CHANGE", f"{change:+.2f} dB", "blue")
    ]) + """
    <h2 class="blue">Original ECG</h2>
    """ + graph_html(original_graph, "Original ECG") + """
    <h2 class="red">Noisy ECG</h2>
    """ + graph_html(noisy_graph, "Noisy ECG") + """
    <h2 class="green">Filtered ECG</h2>
    """ + graph_html(filtered_graph, "Filtered ECG") + f"""
    <div class="card">
        <h2>Interpretation</h2>
        <p>
            SNR before filtering: {before:.2f} dB.
            SNR after filtering: {after:.2f} dB.
        </p>
        <p>
            {"SNR increased, indicating reduced error relative to the reference."
             if change > 0 else
             "SNR did not increase. The filter may need different parameters."}
        </p>
        <p class="small">
            Educational simulation only. Not intended for medical diagnosis.
        </p>
    </div>
    <div class="card">
        <h2>Upload MATLAB Results</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="csv_file" accept=".csv" required>
            <button class="button yellowbtn" type="submit">
                UPLOAD MATLAB CSV
            </button>
        </form>
        <p>{message}</p>
        <a class="button bluebtn" href="/download-data">DOWNLOAD CURRENT CSV</a>
    </div>
    """
    return render_page(
        "Results", body, "results",
        previous="filter_page"
    )


# ---------------- DOWNLOAD CSV ----------------

@app.route("/download-data")
def download_data():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "Clean_ECG", "Noisy_ECG", "Filtered_ECG"])

    for values in zip(time_data, clean_data, noisy_data, filtered_data):
        writer.writerow(values)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=ECG_Results.csv"
        }
    )


# ---------------- RUN WEBSITE ----------------

if __name__ == "__main__":
    generate_signal()
    print("=" * 50)
    print("CARDIOCLEAN WEB DASHBOARD")
    print("Open Chrome at http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host="127.0.0.1", port=5000)
