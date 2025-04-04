# Image Caption Sentiment Analysis API

A lightweight Flask API for performing sentiment analysis using Hugging Face transformer model DistillBERT.


## 📁 Project Structure
    DREAMS/
    ├── dreamsApp/
    │   ├── __init__.py                #        App factory
    │   ├── captionSentiments.py       # API logic and model loading
 

## 🧠 How It Works

The API exposes a single endpoint:

### 📮 POST `/sentiments/caption`

**Request:**

```json
{
  "caption": "I am getting better #recovring"
}
```

**Response**
```json
[
    {'label': 'POSITIVE',
    'score': 0.9030401110649109
    }
]
```
## 💡 Use Case & Integration with Beehive

This API is designed to classify the **sentiment** of a text caption—currently leveraging the powerful `distilBERT` model from Hugging Face. It plays a valuable role in analyzing patient recovery trends by extracting emotional context from captions uploaded alongside photos in Beehive.

### 🧠 The Idea

Whenever a user uploads a **photo** on Beehive and adds a **caption**, that caption is sent to this API:

1. 🔁 Beehive → sends caption to `/sentiments/caption`
2. 🧠 This API → returns sentiment classification
3. 🧾 Beehive → stores this result in the database

Over time, this data allows us to:

- 🔍 Track patient emotional progress (improving, worsening, fluctuating)
- 📊 Visualize trends in mental/emotional recovery
- 🧠 Add valuable metadata to patient records
- 🧪 Enable research on emotional impact of treatments

---

### 🔬 Current Model

- **Model**: `distilBERT-base-uncased-finetuned-sst-2-english`
- **Why it works**: Lightweight, fast, and achieves excellent performance on general sentiment classification tasks.

---

### 🚀 Future Plans

As we gather more **domain-specific data** (medical captions), we aim to:

- 🎯 **Fine-tune** the model on our own dataset
- 🧾 Introduce **custom classes** (e.g., Hopeful, Anxious, Calm, etc.)
- 📈 Improve accuracy through **data augmentation**
- 🧠 Build a robust medical sentiment engine tailored to Beehive

---

### 🤝 Why It Matters

This approach turns passive user input into **valuable insights**. It enables Beehive to go beyond storage and into intelligent analysis—helping caregivers, researchers, and patients themselves.

