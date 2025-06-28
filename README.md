# 🏗️ S-Index for Real Estate Developers

**Course project — HSE, 2025**  
**Authors**: Viktoriia1928, Kirill Budyak  
**Role**: Data Analyst, NLP Engineer

📊 _A complete ML pipeline for evaluating the **social responsibility index (S-index)** of real estate developers based on user reviews from Russian real‑estate platforms._

---

## 📌 Project Goal

To develop an interpretable, data‑driven system that:

* Automatically scrapes and aggregates user reviews from **Avito** and **CIAN**  
* Classifies sentiment using a fine‑tuned **ruBERT‑large** model  
* Extracts key discussion topics through **c‑TF‑IDF**  
* Computes an **S‑index** using weighted sentiment scores for each topic

---

## 🧱 Pipeline Overview

Avito + CIAN Reviews
↓
Web Scraping (Playwright + Asyncio)
↓
Text Cleaning & Lemmatization (Natasha)
↓
Sentiment Classification (ruBERT‑large, F1 ≈ 0.82)
↓
Topic Clustering (TF‑IDF + KMeans)
↓
Topic Modeling (c‑TF‑IDF) → 4 social categories
↓
Logistic Regression → Topic Weights
↓
S‑Index Calculation → Final Developer Ranking
