"""
============================================================
  Project : Engineering College Recommendation System
  Name    : Nishal KV
  Roll No : 67
============================================================

deep_learning_model.py
Implementation of a Neural Network using Keras for college 
suitability prediction.
"""

import os
import silence_tensorflow.auto  # Optional: to keep logs clean
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from recommendation_engine import _normalise, _compute_weighted_score, _assign_labels

def build_nn_model(input_shape, num_classes):
    """Simple Feed-Forward Neural Network for Classification."""
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_deep_learning_model(df, features):
    """Train the Keras model on the provided dataframe."""
    df = df.copy()
    
    # Ensure dataset is labeled for training
    if 'Recommendation' not in df.columns:
        norm = _normalise(df)
        df["Final Score"] = _compute_weighted_score(norm).values
        df["Recommendation"] = _assign_labels(df["Final Score"])
        
    # Convert labels to integers
    le = LabelEncoder()
    y = le.fit_transform(df['Recommendation'])
    X = df[features].values
    
    num_classes = len(le.classes_)
    input_shape = len(features)
    
    model = build_nn_model(input_shape, num_classes)
    
    # Train (briefly for demonstration)
    model.fit(X, y, epochs=20, batch_size=8, verbose=0)
    
    return model, le

def predict_nn(model, le, X_input):
    """Predict labels using the trained NN model."""
    predictions = model.predict(X_input, verbose=0)
    class_indices = np.argmax(predictions, axis=1)
    return le.inverse_transform(class_indices)
