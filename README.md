
# Financial Distress Early Warning System

### End-to-End Bankruptcy Risk Prediction Pipeline

**Author:** Eegunjobi Feranmi | **Dataset:** Polish Companies Bankruptcy (UCI) | **Domain:** FinTech / Risk Analytics

---

## Overview

According to Allianz Trade's Global Insolvency Outlook (April 2026), corporate insolvencies will rise by 6% in 2026, pushing failures to a record high, 24% above pre-pandemic levels and putting 2.2 million jobs directly at risk.

Traditional manual review cannot screen every borrower in time. This project builds a machine learning pipeline that predicts bankruptcy risk from financial ratios, explains predictions using SHAP, and powers a Risk Intelligence Dashboard in Power BI. It ranks companies so analysts can investigate the riskiest ones first, achieving 88.9% recall on the held-out test set.

---

## Model Performance

| Metric | Score |
| :---: | :---: |
| PR AUC | 85.0% |
| Precision | 41.4% |
| Recall | 88.9% |
| F1 Score | 56.5% |
| Train vs Test Gap | 5.0% |
| Overfitting | None confirmed |

> LightGBM outperforms Elastic Net on PR AUC, catching 48 out of 54 failures while flagging only 116 companies, compared to Elastic Net flagging 561 companies to catch 42 failures.

---

## Project Structure

```text
financial-distress-early-warning/
│
├── 01_training/
│   ├── Plots/
│   │   ├── Confusion_matrices_for_both.png
│   │   ├── Risk_ranking.png
│   │   ├── SHAP_beeswarm_plot.png
│   │   ├── SHAP_summary_plot.png
│   │   ├── eda_01_class_imbalance.png
│   │   ├── eda_02_ratio_distribution.png
│   │   ├── eda_03_correlation.png
│   │   ├── eda_04_duplicate_and_conflicting.png
│   │   ├── eda_05_missing_vs_present_bankruptcy.png
│   │   └── eda_06_top_20_correlated_with_bankruptcy.png
│   │
│   ├── Early_Warning_Financial_Distress_Model.ipynb
│   └── high_tier_risk_reports.txt
│
├── 02_deployment/
│   ├── config.py
│   ├── deploy.py
│   ├── etl_pipeline.py
│   └── requirements.txt
│
├── 03_powerbi/
│   └── Financial-Distress-Dashboard.pdf
│
├── athena/
│   └── athena_validation.ipynb
│
├── .gitignore
└── README.md
```

---

## Dataset

### Polish Companies Bankruptcy Dataset

- **Source:** [UCI Machine Learning Repository — Polish Companies Bankruptcy Dataset](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- The dataset provides five separate forecasting datasets. This project uses only the **first-year forecasting file (`1year.arff`)**.
- Contains **7,027 company observations**, including 271 bankrupt companies (3.9%) and 6,756 non-bankrupt companies (96.1%).
- Approximately 25 healthy companies for every bankrupt company.
- Financial ratios are drawn from historical company records, principally covering **2000–2013**.

### Dataset Overview

| Attribute | Description |
|---|---|
| Dataset | Polish Companies Bankruptcy |
| Source | UCI Machine Learning Repository |
| File Used | 1year.arff |
| Total Observations | 7,027 |
| Financial Features | 64 |
| Bankrupt Companies | 271 (3.9%) |
| Non-Bankrupt Companies | 6,756 (96.1%) |
| Class Distribution | Approximately 25:1 |
| Historical Period | Principally 2000–2013 |
| Target Variable | class |
| Forecasting Period | 1 year |

### Financial Attributes Description

The dataset contains 64 financial ratios representing different aspects of corporate financial performance, including profitability, liquidity, leverage, operational efficiency, and financial structure.

The table below provides the description of each financial attribute.

| Column | Description |
|---|---|
| Attr1 | Net profit / total assets |
| Attr2 | Total liabilities / total assets |
| Attr3 | Working capital / total assets |
| Attr4 | Current assets / short-term liabilities |
| Attr5 | Cash, securities and receivables less short-term liabilities, in days of operating expenses |
| Attr6 | Retained earnings / total assets |
| Attr7 | EBIT / total assets |
| Attr8 | Book value of equity / total liabilities |
| Attr9 | Sales / total assets |
| Attr10 | Equity / total assets |
| Attr11 | (Gross profit + extraordinary items + financial expenses) / total assets |
| Attr12 | Gross profit / short-term liabilities |
| Attr13 | (Gross profit + depreciation) / sales |
| Attr14 | (Gross profit + interest) / total assets |
| Attr15 | Total liabilities in days of (gross profit + depreciation) |
| Attr16 | (Gross profit + depreciation) / total liabilities |
| Attr17 | Total assets / total liabilities |
| Attr18 | Gross profit / total assets |
| Attr19 | Gross profit / sales |
| Attr20 | Inventory in days of sales |
| Attr21 | Sales growth (sales this year / sales last year) |
| Attr22 | Operating profit / total assets |
| Attr23 | Net profit / sales |
| Attr24 | Gross profit (in 3 years) / total assets |
| Attr25 | (Equity - share capital) / total assets |
| Attr26 | (Net profit + depreciation) / total liabilities |
| Attr27 | Operating profit / financial expenses |
| Attr28 | Working capital / fixed assets |
| Attr29 | Logarithm of total assets |
| Attr30 | (Total liabilities - cash) / sales |
| Attr31 | (Gross profit + interest) / sales |
| Attr32 | Current liabilities in days of cost of products sold |
| Attr33 | Operating expenses / short-term liabilities |
| Attr34 | Operating expenses / total liabilities |
| Attr35 | Profit on sales / total assets |
| Attr36 | Total sales / total assets |
| Attr37 | (Current assets - inventory) / long-term liabilities |
| Attr38 | Constant capital / total assets |
| Attr39 | Profit on sales / sales |
| Attr40 | (Current assets - inventory - receivables) / short-term liabilities |
| Attr41 | Total liabilities / annualised operating profit plus depreciation |
| Attr42 | Operating profit / sales |
| Attr43 | Receivables and inventory turnover in days |
| Attr44 | Receivables in days of sales |
| Attr45 | Net profit / inventory |
| Attr46 | (Current assets - inventory) / short-term liabilities |
| Attr47 | Inventory in days of cost of products sold |
| Attr48 | EBITDA / total assets |
| Attr49 | EBITDA / sales |
| Attr50 | Current assets / total liabilities |
| Attr51 | Short-term liabilities / total assets |
| Attr52 | Short-term liabilities in days of cost of products sold |
| Attr53 | Equity / fixed assets |
| Attr54 | Constant capital / fixed assets |
| Attr55 | Working capital |
| Attr56 | (Sales - cost of products sold) / sales |
| Attr57 | Liquid current assets less short-term liabilities / operating cost base |
| Attr58 | Total costs / total sales |
| Attr59 | Long-term liabilities / equity |
| Attr60 | Sales / inventory |
| Attr61 | Sales / receivables |
| Attr62 | Short-term liabilities in days of sales |
| Attr63 | Sales / short-term liabilities |
| Attr64 | Sales / fixed assets |

### Target Variable

| Column | Description |
|---|---|
| class | Bankruptcy classification: 1 = Bankrupt, 0 = Non-bankrupt |

**Note:** The dataset contains five forecasting files, but only the first-year forecasting file (`1year.arff`) is used in this project. All analysis and model evaluation results therefore apply exclusively to this file.

