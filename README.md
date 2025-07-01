
# Ancient Greek Inscriptions: Dating and Restoration

This project focuses on the computational analysis of ancient Greek inscriptions, tackling two core tasks using machine learning and evolutionary algorithms:

## 1. Chronological Dating (`ameros.py`)

**Objective:**  
Predict the exact chronological date of ancient Greek inscriptions based on their text content.

**Dataset:**  
- 2,802 inscriptions from the PHI (Packard Humanities Institute) database.
- Each inscription includes its transcribed text and a known date range.

**Approach:**  
- Preprocess texts with tokenization and TF-IDF vectorization.
- Train a neural network model to regress the precise date.
- Evaluate model performance using appropriate regression metrics.

## 2. Restoration (`bmeros.py`)

**Objective:**  
Reconstruct missing or damaged parts of inscriptions.

**Approach:**  
- Apply a Genetic Algorithm (GA) to predict missing words.
- Leverage textual similarity with inscriptions from the same region or time period.
- Use cosine similarity to evaluate the quality of the restorations.

## Summary

This pipeline combines neural networks for chronological prediction and evolutionary algorithms for semantic restoration, contributing to the digital analysis and preservation of ancient texts.


