# Financial Distress Early-Warning System for Corporate Credit Risk

A machine-learning decision-support prototype that analyzes historical corporate financial ratios to identify companies exhibiting elevated bankruptcy risk and help credit analysts prioritize further investigation.

## 1. Business Problem

Banks, lenders, insurers, and investment firms face the challenge of assessing financial risk across large portfolios of companies. Traditional financial reviews can be time-consuming, and warning signs of financial distress may not always receive attention early enough.

Failure to identify deteriorating financial conditions can expose organizations to loan defaults, credit losses, and poor investment decisions.

**Business question:** Can historical financial ratios be used to develop an early-warning model that identifies companies at elevated risk of bankruptcy and helps analysts allocate their investigative resources more effectively?

## 2. Project Objectives

This project investigates five key questions:

1. **Risk detection:** Can historical financial ratios distinguish companies that subsequently experienced bankruptcy from those that did not?
2. **Risk indicators:** Which financial variables contribute most to the model's predictions?
3. **Model performance:** How effectively can the model identify bankrupt companies, and how many cases does it miss?
4. **False alarms:** How many financially healthy companies are incorrectly flagged for investigation?
5. **Business application:** How can model outputs support risk prioritization and further financial review?

## 3. Dataset

**Source:** [Polish Companies Bankruptcy Data — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)

The project uses the first-year dataset file, `1year.arff`, from the UCI Polish Companies Bankruptcy Dataset.

- **Observations:** 7,027 company records, subject to verification during data loading.
- **Predictors:** 64 financial ratios.
- **Target:** Historical bankruptcy classification.
- **Data quality:** The dataset contains missing values that must be assessed and handled appropriately.
- **Class distribution:** Bankruptcy cases are a minority class, making class imbalance a central modeling challenge.

The dataset's filename requires careful interpretation: the `1year.arff` subset uses financial information from the first year of the forecasting period to predict bankruptcy status over a five-year horizon. The prediction horizon will be documented accurately throughout the analysis.

The raw dataset will remain separate from the processed data and will not be modified during the analysis.

## 4. Analytical Approach

The project follows a reproducible machine-learning workflow:

1. **Data auditing and exploratory analysis** — inspect the dataset, missing values, class distribution, and financial-variable distributions.
2. **Data preprocessing** — prepare features, handle missing values, and establish a leakage-resistant validation strategy.
3. **Baseline modeling** — establish a simple reference model before evaluating more advanced approaches.
4. **Model development** — compare suitable classification algorithms, including logistic regression and random forest.
5. **Imbalanced-class evaluation** — assess recall, precision, average precision, and confusion matrices, rather than relying on accuracy alone.
6. **Model interpretation** — investigate which financial indicators influence model predictions.
7. **Risk prioritization** — explore how model scores could help analysts identify companies requiring closer review.
8. **Deployment and reporting** — develop a prototype inference workflow and communicate findings through an analyst-focused dashboard.

## 5. Evaluation Strategy and Business Trade-Offs

Accuracy alone can be misleading when bankruptcy cases represent only a small proportion of observations. A model could achieve high accuracy while failing to identify many distressed companies.

The evaluation will therefore emphasize:

- **Recall:** The proportion of actual bankruptcy cases correctly identified.
- **Precision:** The proportion of companies flagged by the model that are actual bankruptcy cases.
- **Average precision:** A summary measure of the precision-recall trade-off across classification thresholds.
- **Confusion matrix:** A breakdown of correctly identified cases, missed bankruptcy cases, and false alarms.

The operating threshold will be selected using training and validation data, with the final test set reserved for an unbiased performance assessment.

The business assumption is that missing a genuinely distressed company may be more costly than investigating a healthy company. However, the relative costs must be validated with stakeholders rather than treated as an established universal rule.

## 6. Technology Stack

- **Python** — data analysis and model development.
- **Pandas and NumPy** — data manipulation.
- **Scikit-learn** — preprocessing, classification, and evaluation.
- **SciPy** — loading the ARFF dataset.
- **Matplotlib** — exploratory visualizations and evaluation plots.
- **Git and GitHub** — version control, project documentation, and reproducibility.
- **Power BI** — analyst-focused reporting and risk-performance visualization.
- **AWS** — planned prototype deployment, subject to implementation and testing.

## 7. Expected Deliverables

- A documented and reproducible data-analysis workflow.
- A comparison of baseline and machine-learning classification models.
- Evaluation results focused on detecting bankruptcy cases.
- An interpretation of influential financial indicators.
- A prototype risk-scoring workflow for analyst review.
- A dashboard communicating model performance and potential business use.
- Documentation covering assumptions, limitations, and deployment considerations.

All performance figures, model comparisons, and business findings will be reported after the relevant analyses have been completed. No results will be assumed in advance.

## 8. Limitations and Responsible Use

This project uses historical bankruptcy data from Polish companies. Its results cannot automatically be generalized to present-day companies in Nigeria or other markets, where accounting practices, economic conditions, industry composition, and lending environments may differ.

Historical associations in financial ratios do not establish causation. Model predictions may also be affected by missing data, class imbalance, changes in economic conditions, and differences between the original dataset and future applications.

**This is a prototype for decision support, not an automated credit-decision system.** Any real-world application would require independent validation, appropriate governance, monitoring, and review by qualified financial-risk professionals.

## 9. Project Status

**Current stage:** Repository setup and dataset validation.

The project will progress through data exploration, preprocessing, model development, evaluation, interpretation, and prototype deployment. Results and implementation details will be updated as each stage is completed.

## 10. Reproducibility

The repository will contain the analysis code, project documentation, dependency specifications, and tests where appropriate.

The raw dataset will be downloaded from the official UCI source rather than committed to the repository. Instructions for obtaining and preparing the data will be maintained in the project documentation.

---

**Author:** Feranmi Eegunjobi

**Project focus:** Applied data science, imbalanced classification, financial risk analytics, model interpretability, and decision-support systems.