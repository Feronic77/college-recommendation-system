# 🎓 Engineering College Recommendation System

**Name:** Nishal KV  
**Roll No:** 67

## Overview
A Python-based recommendation engine that generates a synthetic benchmark dataset and recommends engineering colleges based on user preferences — preferred branch, tuition budget, and location score.

## Features
- **Synthetic Dataset**: Generates 200 college records across 8 engineering branches.
- **Weighted Scoring**: MinMaxScaler normalization + composite weighted score.
- **RandomForest Model**: Predicts suitability labels using ensemble learning.
- **Deep Learning (NN)**: Keras-based Neural Network for comparative recommendation logic.
- **Generative AI Summary**: Narrative guidance generator for personalized advice.
- **4 Visualizations**: Correlation Heatmap, Distribution charts, and more.
- **Submission Automation**: Script to bundle project and docs into the required `.tar` format.

## Tech Stack
| Library | Purpose |
|---------|---------|
| Pandas | Data handling |
| NumPy | Numerical processing |
| Scikit-learn | RandomForest & Scaling |
| TensorFlow/Keras| Deep Learning Model |
| Matplotlib | Chart generation |
| Seaborn | Statistical visualizations |

## Setup & Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run (interactive)
python main.py

# Run (non-interactive)
python main.py --branch "Computer Science" --budget 8 --location 5
```

## Submission Instructions
1. Ensure all `docs/` and `output/` files are present.
2. Run the packaging script:
   ```bash
   python package_project.py
   ```
3. Submit the generated `.tar` file as per the naming convention.

## Project Structure
```
├── main.py                    # CLI entry point
├── generate_dataset.py        # Synthetic data generation
├── recommendation_engine.py   # Scoring & Generative AI logic
├── deep_learning_model.py     # Keras Neural Network
├── visualizations.py          # Matplotlib/Seaborn charts
├── package_project.py         # Submission packager
├── docs/                      # PPT & Project Report skeletons
├── output/                    # Generated charts
└── requirements.txt           # Dependencies
```
