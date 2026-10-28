# Data Documentation

## 1. Dataset Overview

**Dataset:** Polish Companies Bankruptcy Dataset  
**Source:** UCI Machine Learning Repository  
**Official URL:** https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data  
**Selected file:** `1year.arff`  
**File location:** `data/raw/1year.arff`

This project uses historical financial data from Polish companies to investigate whether financial ratios can help identify companies at elevated risk of bankruptcy.

## 2. Dataset Description

The selected file is expected to contain:

- **Company records:** 7,027 observations.
- **Predictor variables:** 64 financial ratios, identified as `Attr1` through `Attr64`.
- **Target variable:** `class`, representing the bankruptcy classification.
- **Class distribution:** Approximately 3.9% bankruptcy cases, subject to verification against the downloaded file.

The dataset contains missing values and a substantial imbalance between bankruptcy and non-bankruptcy cases. These characteristics will be examined during data validation and exploratory data analysis.

**Target interpretation:** The target labels are expected to distinguish bankruptcy cases (`1`) from non-bankruptcy cases (`0`). The actual column name, label encoding, record count, and class distribution must be confirmed after loading the file.

## 3. Important Note on the Prediction Horizon

The filename `1year.arff` can be misleading. In the UCI dataset, this subset uses financial information from the first year of the forecasting period to predict bankruptcy status over a five-year horizon.

The project will preserve this distinction in its analysis and reporting rather than describing the selected file as a one-year-ahead prediction dataset.

## 4. Why `data/raw/` Is Empty on GitHub

The raw dataset is intentionally excluded from version control.

This keeps the repository lightweight and separates the original data from the code and documentation. The `.gitignore` file should exclude the raw data file without excluding the instructions needed to reproduce the project.

The dataset's reuse and attribution requirements should be followed according to the terms stated by the official UCI source.

## 5. How to Obtain the Data

To reproduce the analysis:

1. Visit the [official UCI dataset page](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data).
2. Download the dataset archive, commonly provided as `data.zip`.
3. Extract the archive locally.
4. Locate the `1year.arff` file.
5. Place the file in the following project directory:

   `data/raw/1year.arff`

6. Run the project's data-loading and validation workflow to confirm the file structure, column names, missing values, and target distribution.

**Expected directory structure:**

```text
data/
├── README.md
├── raw/
│   └── 1year.arff
└── processed/
```

The `processed/` directory will hold derived datasets created during analysis. The original file in `raw/` should remain unchanged.

## 6. Data Quality and Reproducibility

Before modeling, the dataset will be checked for:

- Expected number of observations and variables.
- Correct target-column identification and label encoding.
- Missing and invalid values.
- Duplicate records, where applicable.
- Bankruptcy class distribution.
- Unexpected data types or parsing issues.

Any preprocessing decisions will be documented, and transformations that learn from the data will be fitted using the training data only to reduce the risk of data leakage.

## 7. Limitations

This is a historical benchmark dataset drawn from Polish companies. It does not, by itself, establish that a model trained on these records will accurately predict bankruptcy among modern companies or businesses operating in Nigeria.

Differences in economic conditions, accounting practices, industries, and time periods may affect model performance. Any practical application would require independent validation on relevant data.

The purpose of this project is to demonstrate a reproducible financial-distress modeling workflow and evaluate its usefulness as a decision-support prototype.