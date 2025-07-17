# DREAM Integration: Multimodal Emotion & Sentiment Analysis

This is the ongoing work for a GSoC 2025 project that analyzes emotions and sentiments from image, audio, and text data, where time plays an important role in connecting and understanding the flow of these inputs. 

## Current Status

- Project structure sample data folders, virtual environment, and requirements file has been created.
- Audio transcript generation using [OpenAI Whisper](https://github.com/openai/whisper).
- One test sample added for person-01.Audio to text Conversion done Successfully.

## Tools Used (Till Now)

- Whisper (for audio transcription)
- Python 3.x
- Virtualenv

## Currently Working on

Focusing on fast, local execution of models for audio, text, and image sentiment/emotion analysis using Using pretrained open-source models only.

Once single-sample, tri-modal analysis (image, audio, text) for one user is functional, the next step is to extend the pipeline to handle multiple samples for the same user. This will allow us to observe how emotional indicators vary across different inputs over time. For each sample, we extract emotion or sentiment scores . These scores will then be timestamped and aligned to visualize emotion progression across time, enabling timeline-based analysis per user.

