# DREAMS Analytics Setup Guide

A simple guide to set up and run the DREAMS emotion detection model on your local machine.
The instructions assume you have Python 3.12 or 3.13 installed.
---

## Prerequisites

- **Python 3.12 or 3.13**
- **pip** (Python package manager, comes with Python)
- **Git** (to clone the repository)

---

## Required Dependencies

Before installation, make sure you have these Python packages. They will be installed automatically via `requirements.txt`:

### Core Packages
| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `pandas` | Data handling |
| `Pillow` | Image loading |

### Machine Learning (for Emotion Model)
| Package | Purpose |
|---------|---------|
| `tensorflow` | Deep learning framework |
| `keras` | Neural network API |
| `opencv-python-headless` | Face detection & image processing |

### Web Server
| Package | Purpose |
|---------|---------|
| `Flask` | Web application server |

### Optional (for full features)
| Package | Purpose |
|---------|---------|
| `torch` | PyTorch (alternative ML) |
| `transformers` | HuggingFace models |
| `scikit-learn` | Machine learning utilities |
| `matplotlib` | Visualization |

---

## macOS Setup


### Step 1: Clone and Enter the Project

```bash
cd Desktop/dreams/DREAMS
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install ML model dependencies for Python
pip install tensorflow keras opencv-python-headless pillow
```

### Step 4: Run the Server

```bash
python analytics_server.py
```

### Step 5: Open in Browser

Visit: **http://127.0.0.1:5001**

---

## Windows Setup

### Step 1: Open Command Prompt

Press `Win + R`, type `cmd`, and press Enter.

### Step 2: Navigate to Project

```cmd
cd Desktop\dreams\DREAMS
```

### Step 3: Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 4: Install Dependencies

```cmd
pip install -r requirements.txt
pip install tensorflow keras opencv-python-headless pillow
```

### Step 5: Run the Server

```cmd
python analytics_server.py
```

### Step 5: Open in Browser

Visit: **http://127.0.0.1:5001**

---

## What You'll See

1. **Homepage**: Shows 3 sample users (Alice, Bob, Carol)
2. **Narrative View**: Click a user to see their emotion timeline and episodes
3. **Image Analysis**: Click on images to see real-time emotion detection using the ML model

---

## Troubleshooting

### "Module not found" Error

```bash
pip install <module_name>
```



### Model Loading Error

Ensure the model file exists:
```
latest-model/face_classification/trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5
```

### Port Already in Use

Change the port in `analytics_server.py` (line 763):
```python
app.run(debug=True, port=5002)  # Use a different port
```

---

## Project Structure

```
DREAMS/
├── analytics_server.py      # Main Flask server
├── ml/
│   ├── latest_emotion_model.py   # ML model wrapper
│   └── keras_inference.py        # Keras model loader
├── latest-model/
│   └── face_classification/
│       └── trained_models/
│           └── fer2013_mini_XCEPTION.119-0.65.hdf5  # The emotion model
└── images/                  # Sample images for testing
```

---

## Quick Test

After starting the server, test the emotion API:

```bash
curl http://127.0.0.1:5001/api/perceptual-emotion/download.jpeg
```

You should see a JSON response with emotion probabilities.

---

**Need help?** Check the main README.md or open an issue.
