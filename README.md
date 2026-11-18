# Financial Distress Early Warning System

### End-to-End Bankruptcy Risk Prediction Pipeline

**Author:** Eegunjobi Feranmi | **Dataset:** Polish Companies Bankruptcy (UCI) | **Domain:** FinTech / Risk Analytics

---
## Overview

According to Allianz Trade, Global Insolvency Outlook (April 2026), corporate insolvencies will rise by 6% in 2026, pushing failures to a record high, 24% above levels before the pandemic and putting 2.2 million jobs directly at risk. Traditional manual review cannot screen every borrower in time. This project builds a machine learning pipeline ready for production that automatically predicts bankruptcy risk from financial ratios, explains every decision using SHAP, and powers a Risk Intelligence Dashboard in Power BI, ranking companies so analysts can investigate the riskiest ones first, catching 88.9% of failures while cutting manual screening effort by up to 90%.

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


## Project Structure

<pre>
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

