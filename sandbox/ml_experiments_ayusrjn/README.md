# ML Experiments - DREAMS

## Project Overview
This folder contains machine learning experiments for classifying user recovery journeys based on text captions uploaded to the Beehive platform. The project is built on the CHIME Personal Recovery Framework and uses synthetic datasets for model training and evaluation.

## Objective
- **Primary Goal**: Classify user recovery journey stages from text captions
- **Future Expansion**: Extend classification to include image analysis
- **Framework**: CHIME Personal Recovery Framework
- **Platform**: Beehive platform integration

## Dataset
- **Type**: Synthetic dataset based on CHIME framework
- **Content**:Trying to stimulate user-generated captions from recovery journey posts
- **Size**: Currently 505 Unique Entries Expanding
- **Classes**: Connectedness, Hope, Identity, Meaning, Empowerment
- **Link**: [HuggingFace](https://huggingface.co/datasets/ayusrjn/CHIME-recovery-framework)


## Experiments Status

### Experiment Log

| Experiment ID | Model | Status | Best Accuracy | Best F1 | Notes |
|---------------|-------|--------|---------------|---------|-------|
| EXP-001 | [bert-base-uncased] | In Progress | 0.8431 | 0.8416544 | badly overfitted |
| EXP-002 | [Model Name] | Planned | - | - | - |
| EXP-003 | [Model Name] | Planned | - | - | - |

## Current Model
[HuggingFace Hub](https://huggingface.co/ayusrjn/bert-finetuned-on-chime/tree/main)


## Notes
- This experiment focuses on text-based classification from captions
- Future work will expand to multimodal (text + image) analysis
- All experiments use synthetic data based on CHIME framework principles
- Results will be benchmarked against implementation approach


---
*Last Updated: [05/06/2025]*