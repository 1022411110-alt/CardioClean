
import os
import csv
import base64
from io import BytesIO, StringIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template_string,
    request,
    redirect,
    url_for,
    send_file,
    flash,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "cardioclean-local-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# --------------------------------------------------
# GLOBAL ECG DATA
# --------------------------------------------------

FS = 500
DURATION = 10

time_data = np.array([])
clean_data = np.array([])
noisy_data = np.array([])
filtered_data = np.array([])
data_source = "Built-in simulated ECG"


# --------------------------------------------------
# SIGNAL GENERATION
# --------------------------------------------------

def generate_signal():
    """Generate a simulated ECG signal and add artificial noise."""
    global time_data, clean_data, noisy_data, filtered_data
    global data_source

    time_data = np.arange(0, DURATION, 1 / FS)

    # Create a simulated ECG waveform using Gaussian components.
    clean_data = np.zeros_like(time_data)
    beat_times = np.arange(0.5, DURATION, 1.0)

    for beat in beat_times:
        clean_data += (
            0.12 * np.exp(-((time_data - (beat - 0.20)) / 0.045) ** 2)
        )
        clean_data += (
            -0.15 * np.exp(-((time_data - (beat - 0.045)) / 0.014) ** 2)
        )
        clean_data += (
            1.00 * np.exp(-((time_data - beat) / 0.018) ** 2)
        )
        clean_data += (
            -0.25 * np.exp(-((time_data - (beat + 0.035)) / 0.020) ** 2)
        )
        clean_data += (
            0.30 * np.exp(-((time_data - (beat + 0.24)) / 0.085) ** 2)
        )

    # Add 50 Hz power-line interference and random noise.
    rng = np.random.default_rng(42)
    powerline_noise = 0.12 * np.sin(2 * np.pi * 50 * time_data)
    random_noise = 0.08 * rng.normal(size=len(time_data))
    baseline_wander = 0.08 * np.sin(2 * np.pi * 0.5 * time_data)

    noisy_data = clean_data + powerline_noise + random_noise + baseline_wander
    filtered_data = apply_filter(noisy_data)

    data_source = "Built-in simulated ECG"


# --------------------------------------------------
# DIGITAL FIR LOW-PASS FILTER
# --------------------------------------------------

def apply_filter(signal):
    """Apply an FIR low-pass filter to reduce high-frequency noise."""
    signal = np.asarray(signal, dtype=float)

    if signal.size < 2:
        return signal.copy()

    order = 80
    cutoff = 35.0
    normalized_cutoff = cutoff / (FS / 2)

    taps = np.arange(order + 1) - order / 2
    coefficients = normalized_cutoff * np.sinc(
        normalized_cutoff * taps
    )
    coefficients *= np.hamming(order + 1)
    coefficients /= np.sum(coefficients)

    # Avoid a longer output or edge-size mismatch.
    filtered = np.convolve(signal, coefficients, mode="same")

    return filtered


# --------------------------------------------------
# SIGNAL QUALITY METRICS
# --------------------------------------------------

def calculate_snr(reference, observed):
    """Calculate signal-to-noise ratio in decibels."""
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if reference.size == 0 or observed.size == 0:
        return 0.0

    if reference.size != observed.size:
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
    """Calculate root mean square error."""
    if len(reference) == 0 or len(observed) == 0:
        return 0.0

    return float(np.sqrt(np.mean((reference - observed) ** 2)))


def metric(value, suffix=""):
    """Format a numeric metric for display."""
    return f"{value:.3f}{suffix}"


# --------------------------------------------------
# GRAPH GENERATION
# --------------------------------------------------

def graph_image(x, y, title, color, ylabel="Amplitude"):
    """Create a graph and return it as a base64 image."""
    fig, ax = plt.subplots(figsize=(9, 3.2))

    ax.plot(x, y, color=color, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)

    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return "data:image/png;base64," + encoded


def graph_html(x, y, title, color, ylabel="Amplitude"):
    """Return a graph image as HTML."""
    image = graph_image(x, y, title, color, ylabel)
    return f'<img src="{image}" alt="{title}" style="width:100%;border-radius:8px;">'


# --------------------------------------------------
# COMMON PAGE TEMPLATE
# --------------------------------------------------

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} | CardioClean</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            color: #17233c;
        }
        header {
            background: #102b50;
            color: white;
            padding: 18px 6%;
        }
        header h1 { margin: 0 0 8px; }
        nav { display: flex; flex-wrap: wrap; gap: 18px; }
        nav a {
            color: white;
            text-decoration: none;
            font-size: 14px;
        }
        nav a:hover { text-decoration: underline; }
        main {
            max-width: 1100px;
            margin: 28px auto;
            padding: 0 18px;
        }
        .card {
            background: white;
            padding: 22px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 3px 14px rgba(20, 40, 70, 0.07);
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px;
        }
        .number { font-size: 28px; font-weight: bold; color: #1677c8; }
        .button {
            display: inline-block;
            background: #1677c8;
            color: white;
            border: 0;
            padding: 11px 17px;
            margin: 5px 5px 5px 0;
            border-radius: 7px;
            text-decoration: none;
            cursor: pointer;
            font-size: 14px;
        }
        .button:hover { background: #105c9b; }
        .graph { width: 100%; overflow: hidden; }
        .graph img { width: 100%; height: auto; }
        input[type=file] { margin: 12px 0; max-width: 100%; }
        .message { padding: 12px; background: #fff0f0; border-radius: 6px; }
        .muted { color: #617086; }
        footer { text-align: center; padding: 25px; color: #617086; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #e5eaf0; }
        @media(max-width: 600px) {
            header { padding: 18px; }
            nav { gap: 12px; }
            main { margin-top: 18px; }
            .card { padding: 16px; }
        }
    </style>
</head>
<body>
<header>
    <h1>CardioClean</h1>
    <div style="margin-bottom:14px;">ECG Signal Noise Removal System</div>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('signal') }}">Signal</a>
        <a href="{{ url_for('noise') }}">Noise</a>
        <a href="{{ url_for('filter_page') }}">Filtering</a>
        <a href="{{ url_for('results') }}">Results</a>
        <a href="{{ url_for('download_data') }}">Download CSV</a>
    </nav>
</header>

<main>
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            {% for message in messages %}
                <div class="card message">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}

    {{ content|safe }}
</main>

<footer>
    CardioClean · Educational ECG signal processing project
</footer>
</body>
</html>
"""


def render_page(title, content):
    return render_template_string(
        PAGE,
        title=title,
        content=content
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    content = f"""
    <div class="card">
        <h2>Welcome to CardioClean</h2>
        <p>
            CardioClean demonstrates digital signal processing by generating
            an ECG signal, adding simulated noise, and applying a digital
            low-pass FIR filter.
        </p>
        <p class="muted">
            Current data source: {data_source}
        </p>
        <a class="button" href="{url_for('signal')}">View ECG Signal</a>
        <a class="button" href="{url_for('filter_page')}">View Filtered Signal</a>
        <a class="button" href="{url_for('results')}">View Results</a>
    </div>

    <div class="cards">
        <div class="card">
            <h3>Signal Generation</h3>
            <p>Create a simulated ECG waveform.</p>
        </div>
        <div class="card">
            <h3>Noise Addition</h3>
            <p>Demonstrate power-line interference, baseline wander and random noise.</p>
        </div>
        <div class="card">
            <h3>Digital Filtering</h3>
            <p>Use an FIR low-pass filter to reduce high-frequency noise.</p>
        </div>
        <div class="card">
            <h3>Result Analysis</h3>
            <p>Compare the original, noisy and filtered signals.</p>
        </div>
    </div>

    <div class="card">
        <h2>Upload ECG data</h2>
        <p>Upload a CSV file containing your ECG samples.</p>
        <form action="{url_for('upload_csv')}" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <br>
            <button class="button" type="submit">Upload CSV</button>
        </form>
        <p class="muted">
            Supported columns: Time, Clean_ECG, Noisy_ECG, Filtered_ECG.
            The Time column is optional.
        </p>
    </div>
    """
    return render_page("Home", content)


# --------------------------------------------------
# SIGNAL PAGE
# --------------------------------------------------

@app.route("/signal")
def signal():
    if time_data.size == 0:
        generate_signal()

    duration = float(time_data[-1] - time_data[0]) if time_data.size > 1 else 0

    clean_graph = graph_html(
        time_data, clean_data, "Original ECG Signal", "#1677c8"
    )

    content = f"""
    <div class="card">
        <h2>Original ECG Signal</h2>
        <p>View the clean simulated ECG waveform before adding noise.</p>
        <div class="graph">{clean_graph}</div>
        <div class="cards">
            <div><p class="muted">Sampling frequency</p><div class="number">{FS} Hz</div></div>
            <div><p class="muted">Duration</p><div class="number">{duration:.2f} s</div></div>
            <div><p class="muted">Samples</p><div class="number">{len(time_data)}</div></div>
        </div>
        <a class="button" href="{url_for('noise')}">Continue to Noise Analysis</a>
    </div>
    """
    return render_page("Signal", content)


# --------------------------------------------------
# NOISE PAGE
# --------------------------------------------------

@app.route("/noise")
def noise():
    if time_data.size == 0:
        generate_signal()

    clean_graph = graph_html(
        time_data, clean_data, "Clean ECG", "#1677c8"
    )
    noisy_graph = graph_html(
        time_data, noisy_data, "ECG with Added Noise", "#d9534f"
    )

    noisy_snr = calculate_snr(clean_data, noisy_data)

    content = f"""
    <div class="card">
        <h2>Noise Analysis</h2>
        <p>Compare the original ECG with the signal containing simulated noise.</p>
        <h3>Original ECG</h3>
        <div class="graph">{clean_graph}</div>
        <h3>Noisy ECG</h3>
        <div class="graph">{noisy_graph}</div>
        <div class="cards">
            <div><p class="muted">SNR before filtering</p><div class="number">{noisy_snr:.2f} dB</div></div>
            <div><p class="muted">Noise types</p><strong>Power-line, random and baseline wander</strong></div>
        </div>
        <a class="button" href="{url_for('filter_page')}">Apply Filter</a>
    </div>
    """
    return render_page("Noise Analysis", content)


# --------------------------------------------------
# FILTER PAGE
# --------------------------------------------------

@app.route("/filter")
def filter_page():
    if time_data.size == 0:
        generate_signal()

    clean_graph = graph_html(
        time_data, clean_data, "Original ECG", "#1677c8"
    )
    noisy_graph = graph_html(
        time_data, noisy_data, "Noisy ECG", "#d9534f"
    )
    filtered_graph = graph_html(
        time_data, filtered_data, "Filtered ECG", "#19966b"
    )

    before_snr = calculate_snr(clean_data, noisy_data)
    after_snr = calculate_snr(clean_data, filtered_data)

    content = f"""
    <div class="card">
        <h2>Digital Filtering</h2>
        <p>
            The FIR low-pass filter uses a 35 Hz cutoff to reduce high-frequency
            noise. Filtering cannot remove every type of noise perfectly.
        </p>
        <h3>1. Original ECG</h3>
        <div class="graph">{clean_graph}</div>
        <h3>2. Noisy ECG</h3>
        <div class="graph">{noisy_graph}</div>
        <h3>3. Filtered ECG</h3>
        <div class="graph">{filtered_graph}</div>
        <div class="cards">
            <div><p class="muted">SNR before filtering</p><div class="number">{before_snr:.2f} dB</div></div>
            <div><p class="muted">SNR after filtering</p><div class="number">{after_snr:.2f} dB</div></div>
            <div><p class="muted">Filter type</p><div class="number">FIR</div></div>
            <div><p class="muted">Cutoff frequency</p><div class="number">35 Hz</div></div>
        </div>
        <a class="button" href="{url_for('results')}">View Final Results</a>
    </div>
    """
    return render_page("Filtering", content)


# --------------------------------------------------
# RESULTS PAGE
# --------------------------------------------------

@app.route("/results")
def results():
    if time_data.size == 0:
        generate_signal()

    before_snr = calculate_snr(clean_data, noisy_data)
    after_snr = calculate_snr(clean_data, filtered_data)

    before_rmse = calculate_rmse(clean_data, noisy_data)
    after_rmse = calculate_rmse(clean_data, filtered_data)

    improvement = after_snr - before_snr

    clean_graph = graph_html(
        time_data, clean_data, "Clean ECG", "#1677c8"
    )
    noisy_graph = graph_html(
        time_data, noisy_data, "Noisy ECG", "#d9534f"
    )
    filtered_graph = graph_html(
        time_data, filtered_data, "Filtered ECG", "#19966b"
    )

    content = f"""
    <div class="card">
        <h2>ECG Processing Results</h2>
        <p>Review the measurements and compare the three signal stages.</p>
        <div class="cards">
            <div><p class="muted">SNR before filtering</p><div class="number">{before_snr:.2f} dB</div></div>
            <div><p class="muted">SNR after filtering</p><div class="number">{after_snr:.2f} dB</div></div>
            <div><p class="muted">SNR change</p><div class="number">{improvement:+.2f} dB</div></div>
            <div><p class="muted">RMSE before filtering</p><div class="number">{before_rmse:.4f}</div></div>
            <div><p class="muted">RMSE after filtering</p><div class="number">{after_rmse:.4f}</div></div>
        </div>
    </div>

    <div class="card">
        <h3>Signal comparison</h3>
        <div class="graph">{clean_graph}</div>
        <div class="graph">{noisy_graph}</div>
        <div class="graph">{filtered_graph}</div>
    </div>

    <div class="card">
        <h3>Download your results</h3>
        <p>Export the signal samples to a CSV file.</p>
        <a class="button" href="{url_for('download_data')}">Download ECG Results CSV</a>
    </div>

    <div class="card">
        <p class="muted">
            These are educational signal-processing measurements on simulated
            or uploaded data. They are not a medical diagnosis.
        </p>
    </div>
    """
    return render_page("Results", content)


# --------------------------------------------------
# CSV UPLOAD
# --------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_csv():
    global time_data, clean_data, noisy_data, filtered_data, data_source

    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a CSV file first.")
        return redirect(url_for("home"))

    if not uploaded_file.filename.lower().endswith(".csv"):
        flash("Please upload a .csv file.")
        return redirect(url_for("home"))

    try:
        content = uploaded_file.read().decode("utf-8-sig")
        reader = csv.DictReader(StringIO(content))

        if not reader.fieldnames:
            raise ValueError("The CSV file has no header row.")

        # Match column names without depending on capitalization.
        columns = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name
        }

        def get_column(row, possible_names):
            for name in possible_names:
                actual_name = columns.get(name.lower())
                if actual_name is not None:
                    value = row.get(actual_name, "")
                    if value is not None and value.strip() != "":
                        return float(value)
            return None

        rows = []
        for row in reader:
            t = get_column(row, ["Time", "time_seconds", "timestamp"])
            clean = get_column(row, ["Clean_ECG", "clean", "clean_ecg", "original_ecg"])
            noisy = get_column(row, ["Noisy_ECG", "noisy", "noisy_ecg"])
            filtered = get_column(row, ["Filtered_ECG", "filtered", "filtered_ecg"])

            # A CSV containing a clean signal alone can also be used.
            if clean is None:
                clean = get_column(row, ["ECG", "Signal", "Amplitude", "Value"])

            if clean is None:
                continue

            rows.append((t, clean, noisy, filtered))

        if len(rows) < 2:
            raise ValueError(
                "No usable ECG samples found. Include Clean_ECG or ECG values."
            )

        clean_values = np.array([r[1] for r in rows], dtype=float)

        if all(r[0] is not None for r in rows):
            new_time = np.array([r[0] for r in rows], dtype=float)
        else:
            new_time = np.arange(len(rows), dtype=float) / FS

        if all(r[2] is not None for r in rows):
            new_noisy = np.array([r[2] for r in rows], dtype=float)
        else:
            rng = np.random.default_rng(42)
            new_noisy = (
                clean_values
                + 0.12 * np.sin(2 * np.pi * 50 * new_time)
                + 0.08 * rng.normal(size=len(clean_values))
            )

        if all(r[3] is not None for r in rows):
            new_filtered = np.array([r[3] for r in rows], dtype=float)
        else:
            new_filtered = apply_filter(new_noisy)

        time_data = new_time
        clean_data = clean_values
        noisy_data = new_noisy
        filtered_data = new_filtered
        data_source = "Uploaded CSV"

        flash(f"Successfully loaded {len(rows)} ECG samples.")
        return redirect(url_for("results"))

    except (ValueError, UnicodeDecodeError, csv.Error) as error:
        flash(f"Could not read the CSV file: {error}")
        return redirect(url_for("home"))


# --------------------------------------------------
# CSV DOWNLOAD
# --------------------------------------------------

@app.route("/download-data")
def download_data():
    if time_data.size == 0:
        generate_signal()

    output = StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Time",
        "Clean_ECG",
        "Noisy_ECG",
        "Filtered_ECG"
    ])

    for i in range(len(time_data)):
        writer.writerow([
            time_data[i],
            clean_data[i],
            noisy_data[i],
            filtered_data[i]
        ])

    csv_bytes = BytesIO(output.getvalue().encode("utf-8"))
    csv_bytes.seek(0)

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name="ECG_Results.csv"
    )


# --------------------------------------------------
# STARTUP
# IMPORTANT: Generate data before any page needs it.
# --------------------------------------------------

generate_signal()


# --------------------------------------------------
# LOCAL DEVELOPMENT
# Render uses its configured Gunicorn start command.
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
