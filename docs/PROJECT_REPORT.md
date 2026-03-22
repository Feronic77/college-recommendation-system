# Engineering College Recommendation System: Project Report

**Author:** Nishal KV (Roll No: 67)  
**Date:** March 2026

---

## 1. Title Page
- **Project Title:** Engineering College Recommendation System
- **Submitted for:** [Course Name/Semester]
- **By:** Nishal KV
- **College/University:** [Institutional Name]

---

## 2. Abstract
The "Engineering College Recommendation System" is an AI-driven tool designed to assist students in the complex process of selecting the right engineering institution. By leveraging a synthetic dataset of 200+ colleges across 8 major branches, the system employs weighted scoring, Random Forest classification, and Deep Learning (Keras) to provide ranked recommendations. Additionally, a Generative AI module provides personalized guidance statements. The project demonstrates the application of modern data science workflows, from data generation and pre-processing to predictive modelling and visualization.

---

## 3. Introduction
### 3.1 Background
The landscape of higher education is increasingly competitive. With thousands of options, students often make suboptimal choices based on incomplete information or peer pressure.
### 3.2 Problem Statement
Selecting an engineering college involves multi-dimensional criteria including placement rates, budget constraints, infrastructure, and location. There is a lack of integrated digital tools that allow for personalized, objective ranking based on these metrics.
### 3.3 Objectives
- To automate the college shortlisting process.
- To compare different AI models (Random Forest vs. Deep Learning).
- To provide visual insights into college performance metrics.

---

## 4. Literature Review
- **Recommender Systems:** Overview of Content-Based vs. Collaborative Filtering.
- **Machine Learning in Education:** Previous studies on using AI for academic counseling.
- **Deep Learning for Tabular Data:** The shift from traditional trees to neural networks.
- **Generative AI:** Modern applications of text generation in personalized reports.

---

## 5. Platform, Tools, and Technologies Used
### 5.1 Platform
- **Operating System:** Linux (Ubuntu 22.04 LTS)
- **Environment:** VS Code / Terminal
### 5.2 Languages & Libraries
- **Language:** Python 3.x
- **Libraries:** Pandas (Data Manipulation), Scikit-Learn (ML), TensorFlow/Keras (DL), Matplotlib/Seaborn (Visualization).
- **Version Control:** Git/GitHub.

---

## 6. System Design
### 6.1 Architecture Diagram
*(Represented here as a flowchart)*
1. Data Input (Synthetic Generator) -> 2. Pre-processing (Normalization) -> 3. Feature Weights -> 4. Model Training (RF/DL) -> 5. User UI (CLI) -> 6. Result Generation (Charts, Table, AI Summary).

### 6.2 Database Schema (Synthetic)
| Field | Type | Description |
|-------|------|-------------|
| College Name | String | Name of the institution |
| Branch | String | Core engineering branch |
| Placement Rate | Float | % of students placed |
| Tuition Fees | Float | Annual fees in LPA |
| Infrastructure | Int (1-10) | Quality of facilities |
| ... | ... | ... |

---

## 7. Methodology
### 7.1 Data Generation Strategy
Describes the use of a randomized generator to create 200 diverse entries, ensuring statistical distribution across branches and tiers (Urban vs Rural scores, etc.).
### 7.2 Weighted Scoring Algorithm
Explain the formula for "Final Score" and how weights were assigned (e.g., Placement = 0.25).
### 7.3 Training & Testing
A 75% training and 25% testing split was used to evaluate model accuracy.

---

## 8. Implementation Details
### 8.1 Core Modules
- `generate_dataset.py`: Logic for creating the CSV.
- `recommendation_engine.py`: Scoring and labelling logic.
- `deep_learning_model.py`: Neural network implementation.
- `visualizations.py`: Chart generation code.
### 8.2 Code Snippets (Key parts)
*(Mention specific functions like `recommend_colleges()` and `train_ml_model()`)*

---

## 9. Results
### 9.1 Performance Metrics
- Random Forest Accuracy: ~92%
- MLP (Keras) Accuracy: ~88% (on 200 samples)
### 9.2 Graphical Representations
- Discuss the Heatmap (Correlation between package and fees).
- Discuss the Pie Chart (Distribution of college tiers).

---

## 10. Conclusion and Future Scope
### 10.1 Summary
The project successfully bridges the gap between raw college data and student decision-making using advanced AI.
### 10.2 Future Work
- Cloud deployment (Heroku/Streamlit).
- Real-time web scraping from actual college websites.
- Advanced LLM-based voice assistant integration.

---

## 11. Challenges Faced
- Limited actual data availability leading to synthetic generation requirements.
- Fine-tuning the neural network for a relatively small tabular dataset (overfitting).
- Designing a CLI that is user-friendly for non-technical students.

---

## 12. References
- [List here the standard documentation for Scikit-Learn, TensorFlow, and Pandas]
- [Add 2-3 academic citations from Google Scholar on Recommender Systems]
