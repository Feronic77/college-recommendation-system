# 🎓 Engineering College Recommendation System

**Name:** Nishal KV  
**Roll No:** 67

## Overview
A Python-based recommendation engine that generates a synthetic benchmark dataset and recommends engineering colleges based on user preferences — preferred branch, tuition budget, and location score.

## Features
- **Synthetic Dataset**: Generates 200 college records across 8 engineering branches
- **Weighted Scoring**: MinMaxScaler normalization + composite weighted score
- **ML Model**: RandomForest classifier for label prediction
- **4 Visualizations**: Bar chart, Pie chart, Correlation Heatmap, Box plot
- **Recommendation Labels**: Highly Recommended, Recommended, Moderately Suitable, Not Recommended

## Tech Stack
| Library | Purpose |
|---------|---------|
| Pandas | Data handling |
| NumPy | Numerical processing |
| Scikit-learn | MinMaxScaler, RandomForest |
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

## Project Structure
```
├── main.py                    # CLI entry point
├── generate_dataset.py        # Synthetic data generation
├── recommendation_engine.py   # Scoring, labelling & ML model
├── visualizations.py          # Matplotlib/Seaborn charts
├── requirements.txt           # Dependencies
└── output/                    # Generated charts (after running)
```

## Sample Output
The system generates:
- A ranked table of recommended colleges
- Summary statistics
- 4 charts saved to the `output/` folder
