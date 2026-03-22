# Project Submission: PPT Slides Breakdown

This document provides the text and structure for the mandatory **15-slide presentation**.

---

### Slide 1: Title
**Title:** Engineering College Recommendation System  
**Subtitle:** A Machine Learning and Generative AI Approach  
**Presented by:** Nishal KV (Roll No: 67)  
**Date:** March 2026

---

### Slide 2: Introduction
- **Goal:** To help aspiring engineering students find the most suitable colleges based on their academic and personal preferences.
- **Problem:** With thousands of colleges, students often struggle to weigh factors like placement rates, fees, infrastructure, and faculty quality.
- **Solution:** A data-driven system that provides ranked recommendations and personalized guidance.

---

### Slide 3: Problem Statement
- Manual selection of colleges is prone to bias and information overload.
- Lack of a centralized tool that combines quantitative metrics (scores) with qualitative guidance.
- Need for a system that can predict suitability labels (Highly Recommended, etc.) using historical or synthetic data.

---

### Slide 4: Objectives
1. Develop a synthetic dataset of 200+ college records for benchmarking.
2. Implement a weighted scoring algorithm for multi-criteria decision making.
3. Build a Machine Learning classifier (Random Forest) to predict recommendation labels.
4. Integrate a Deep Learning model (Neural Network) for comparative analysis.
5. Provide a Generative AI summary for personalized student advice.

---

### Slide 5: Scope of the Work
- **Target Audience:** Engineering aspirants and educational consultants.
- **Parameters Covered:** Placement %, Avg Package, Fees, Infrastructure, Faculty, and Student Satisfaction.
- **Technologies:** Python, Scikit-learn, TensorFlow/Keras, and Matplotlib.

---

### Slide 6: Technologies Used
- **Language:** Python 3.x
- **Core ML:** Scikit-learn (Random Forest)
- **Deep Learning:** TensorFlow & Keras
- **Data Handling:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Documentation:** Git/GitHub, Markdown

---

### Slide 7: System Design (Architecture)
- **Data Layer:** Synthetic Generator (CSV format).
- **Processing Layer:** Pre-processing (MinMax Scaling), Weighted Scoring.
- **Model Layer:** RF Classifier + Deep Neural Network.
- **UI/Output Layer:** CLI-based interactive prompts + PDF/Chart outputs.

---

### Slide 8: Methodology
1. **Data Generation:** Randomised but realistic data based on industry averages.
2. **Feature Engineering:** Calculating composite scores from 6 key metrics.
3. **Training:** 75-25 split for training and testing ML/DL models.
4. **Recommendation:** Hybrid approach using both filtered scoring and predictive modelling.

---

### Slide 9: Implementation (Data & Scoring)
- **Scoring Formula:** $Score = \sum (Weight_i \times Normalized\_Metric_i)$
- **Weights:** Placement (25%), Package (20%), Faculty (20%), etc.
- **Labelling:** Quantile-based categorisation into 4 tiers.

---

### Slide 10: Implementation (Machine Learning)
- **Random Forest:** 120 estimators, balanced class weights.
- **Evaluation:** High accuracy in predicting tiers based on quantitative features.
- **Model Persistence:** Training occurs on-the-fly or can be saved.

---

### Slide 11: Output Screenshots (1/2)
- *(Note: Capture a screenshot of the CLI table output here)*
- Showing ranked results for "Computer Science" colleges.
- Displays calculated Final Scores and ML Labels.

---

### Slide 12: Output Screenshots (2/2)
- *(Note: Capture screenshots of the 4 generated charts here)*
- Feature Correlation Heatmap.
- Distribution of Recommendation Tiers.
- Rank vs Placement Rate scatter plot.

---

### Slide 13: Challenges Faced
1. Balancing weights to ensure "Highly Recommended" colleges are truly elite.
2. Handling edge cases where specific branches have very few entries.
3. Simulating realistic "Generative" summaries without heavy API dependencies.
4. Designing a robust Deep Learning architecture for small datasets.

---

### Slide 14: Result and Conclusion
- Successfully developed a functional, multi-model recommendation engine.
- The system provides accurate and actionable insights for students.
- Proven that ML/DL can effectively automate the college shortlisting process.

---

### Slide 15: Future Scope
- Integration with live national ranking data (NIRF).
- Web-based interactive dashboard (React/Next.js).
- Real-time LLM integration for conversational counselling.

---

### Slide 16: References
1. Scikit-learn Documentation: "Ensemble Methods".
2. TensorFlow Keras API Guides.
3. Pandas/NumPy user manuals.
4. Academic papers on Hybrid Recommendation Systems.
