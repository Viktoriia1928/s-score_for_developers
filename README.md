# 🏗️ S-Index for Real Estate Developers

**Course project — HSE, 2025**  
**Authors**: Viktoriia Korableva, Kirill Budyak  
**Roles**: Data Analyst, ML-Engineer

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

graph TD
    A[Avito & CIAN<br>Reviews] --> B[Scraping<br>Playwright&nbsp;+&nbsp;Asyncio]
    B --> C[Lemmatization<br>Natasha]
    C --> D[Sentiment Model<br>ruBERT-large<br>(F1≈0.82)]
    D --> E[Vectorisation<br>TF-IDF]
    E --> F[Clustering<br>K-Means (K=6)]
    F --> G[Topic Keywords<br>c-TF-IDF<br>→ 4 social categories]
    G --> H[Logistic Regression<br>Topic Weights]
    H --> I[S-Index Calculation]
    I --> J[Final Developer Ranking]
    classDef box fill:#F8F9FA,stroke:#0d47a1,color:#0d47a1,stroke-width:1px;
    class A,B,C,D,E,F,G,H,I,J box

---

## 📂 Repository Structure

| Filename                                   | Purpose                                                        |
|--------------------------------------------|----------------------------------------------------------------|
| `01_scrape_avito_reviews.ipynb`            | Scraper for Avito user reviews (BeautifulSoap)                 |
| `02_scrape_cian_reviews.ipynb`             | Scraper for CIAN user reviews (Playwright + Asyncio)           |
| `03_merge_scraped_reviews.ipynb`           | Combine, deduplicate and clean reviews                         |
| `04_sentiment_classification_ruBERT.ipynb` | ruBERT‑based sentiment classification pipeline                 |
| `05_topic_modeling_and_s_index.ipynb`      | Clustering, topic modeling and final S‑index calculation       |
| `README.md`                                | Project overview                                               |

---

## 🧠 Sentiment Analysis

We fine‑tuned **`ruBERT‑large`** from SberDevices to classify reviews as:

* **Negative**
* **Neutral**
* **Positive**

**Pre‑processing**: lemmatization (Natasha), tokenization, truncation  
**Training strategy**  
* Optimizer: `AdamW` + weight decay  
* Scheduler: warm‑up steps → linear decay  
* Class imbalance: `Focal Loss` + `WeightedSampler`  
* Metric: macro‑F1 (≈ **0.84**)  
* Hyper‑parameter tuning: **Optuna**, 5‑Fold stratified CV

---

## 🧵 Topic Modeling (c‑TF‑IDF)

We grouped user reviews into interpretable **social‑responsibility topics**:

* 👥 Customer Interaction  
* 🏘️ Housing Quality  
* 🏞️ Local Infrastructure Impact  
* 🛠️ Labor Conditions  

**Steps**

1. Vectorization: `TF‑IDF`  
2. Clustering: `KMeans`, elbow-method _K_ selection  
3. Topic modeling: **c‑TF‑IDF** (class‑based TF‑IDF for interpretability)  
4. Weighting: `LogisticRegression` to estimate each topic’s contribution to the overall sentiment → used as topic weight  

---

## 🧮 S‑Index Calculation
\[
S \;=\;\Bigl(\;\sum_{i=1}^{n} w_i \cdot P_i\Bigr)\times100
\]

* \(w_i\) — weight of topic *i* (from logistic regression)  
* \(P_i\) — percentage of **positive** mentions in topic *i*

The resulting **S‑index** reflects overall **customer‑perceived social responsibility**.

---

## 🛠️ Stack

| Stage              | Tech                                               |
|--------------------|----------------------------------------------------|
| Web scraping       | Playwright, Asyncio**, Beautifulsoap               |
| NLP preprocessing  | Natasha, sentence_transformers                     |
| Classification     | ruBERT‑large, PyTorch, 🤗 Transformers, Optuna     |
| Clustering & topics| Scikit‑learn (K-Means/HDBSCAN ), c‑TF‑IDF, UMAP    |
| S-index calculation| sklearn (LogisticRegression, LinearRegression, OHE)|
| Collaboration      | Git, GitHub, Jupyter Notebook                      |

---

## 📎 The best 10 developers by S-index

| # | Developer            | **S-Score** | Avg. Rating | Houses Built | Estate Segment(s)              | Avg. Price *(RUB / m²)* | Years on Market |
|---|----------------------|------------:|------------:|-------------:|--------------------------------|------------------------:|----------------:|
| 1 | October Group        | **5.00**    | 4.91        | 0            | Business                       | 637 960                | 3               |
| 2 | FORMA                | 4.14        | 4.84        | 5            | Premium                        | 1 033 485              | 4               |
| 3 | Hals-Development     | 4.08        | 4.72        | 35           | Business                       | 261 663                | 31              |
| 4 | Element              | 4.00        | 4.83        | 3            | Business; Premium              | 607 759                | 8               |
| 5 | DONSTROY             | 4.00        | 4.70        | 137          | Comfort; Business; Premium     | 837 629                | 31              |


---

## 🔗 Repository

<https://github.com/Viktoriia1928/s-score_for_developers>

---

## 🏁 Results

* ✔️ Fully automated & reproducible pipeline  
* ✔️ NLP stack (ruBERT + c‑TF‑IDF)  
* ✔️ Actionable ESG‑style metric derived from real user feedback  

---

## 📬 Contacts

* ✉️ budyak.kirill@edu.hse.ru  
* GitHub: <https://github.com/Viktoriia1928>
