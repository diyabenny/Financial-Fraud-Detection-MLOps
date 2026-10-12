# 1.1 Business Goal & Success Metrics

## 1.1.1 Business Context

Financial institutions process a large volume of transactions across digital and physical payment channels. Although most transactions are legitimate, a small proportion may be fraudulent, potentially resulting in financial losses, operational risks, and negative customer impact.

Traditional rule-based fraud detection systems rely on predefined rules, thresholds, and known fraud patterns. While these approaches can be effective for known patterns, they may be difficult to maintain as fraudulent behavior evolves. Static rules may also generate unnecessary alerts for legitimate transactions, increasing the workload of fraud investigation teams.

Therefore, this project aims to develop an automated machine-learning-based financial fraud detection system that can identify potentially fraudulent transactions while reducing unnecessary alerts for legitimate transactions.

---

## 1.1.2 Business Problem

The primary business problem is to reliably identify fraudulent financial transactions within a large volume of legitimate transactions.

The system must consider two major types of classification errors:

### False Negatives

A false negative occurs when a fraudulent transaction is incorrectly classified as legitimate.

Undetected fraudulent transactions may result in:

- Financial losses
- Increased exposure to fraudulent activity
- Negative customer impact
- Reduced effectiveness of fraud monitoring

Therefore, minimizing missed fraudulent transactions is an important business objective.

### False Positives

A false positive occurs when a legitimate transaction is incorrectly classified as fraudulent.

Excessive false positives may result in:

- Unnecessary fraud alerts
- Increased investigation workload
- Additional operational effort
- Customer inconvenience
- Reduced efficiency of fraud investigation teams

Therefore, the system must also control unnecessary alerts while maintaining effective fraud detection.

---

## 1.1.3 Business Goal

The primary business goal of this project is:

> **To develop an automated and reliable machine-learning-based financial fraud detection system that effectively identifies potentially fraudulent transactions while minimizing unnecessary alerts for legitimate transactions.**

The system is intended to support fraud monitoring and investigation activities by:

1. Identifying potentially fraudulent transactions from large volumes of transaction data.
2. Reducing the number of fraudulent transactions incorrectly classified as legitimate.
3. Controlling unnecessary alerts generated for legitimate transactions.
4. Supporting consistent and data-driven transaction risk classification.
5. Helping prioritize suspicious transactions for further investigation.
6. Providing a foundation for integration into an MLOps-based fraud detection workflow.

---

## 1.1.4 Business Priorities

The project has the following business priorities.

### Priority 1 — Effective Fraud Detection

The system should identify a high proportion of fraudulent transactions.

Missing fraudulent transactions can have significant financial and operational consequences. Therefore, the ability to detect fraudulent transactions is a primary business concern.

### Priority 2 — Reduction of Unnecessary Alerts

The system should avoid unnecessarily flagging legitimate transactions as fraudulent.

Reducing false alerts can help decrease investigation workload and improve operational efficiency.

### Priority 3 — Balance Between Fraud Detection and Alert Quality

The system should achieve an appropriate balance between detecting fraudulent transactions and avoiding excessive false alerts.

A model that detects almost every transaction as fraud would not be operationally useful, while a model that rarely identifies fraud would fail to meet the primary business objective.

### Priority 4 — Appropriate Handling of Fraud Imbalance

Fraudulent transactions generally represent only a small proportion of total transactions.

Therefore, overall classification accuracy should not be treated as the sole indicator of business success. The project must evaluate the model using measures that reflect the ability to detect fraud and control false alerts.

### Priority 5 — Operational Readiness

The fraud detection solution should be suitable for integration into an automated MLOps workflow.

Operational considerations such as prediction efficiency, reliability, monitoring, and future model improvement will be addressed during the later stages of the project.

---

## 1.1.5 Business-Level Success Criteria

The project will be considered successful when the developed fraud detection system:

- Identifies fraudulent transactions with meaningful predictive performance.
- Detects a high proportion of actual fraudulent transactions.
- Maintains an acceptable level of false alerts on legitimate transactions.
- Provides a reasonable balance between fraud detection and false-positive reduction.
- Demonstrates consistent performance on previously unseen transaction data.
- Supports data-driven prioritization of potentially fraudulent transactions.
- Can be integrated into the planned MLOps workflow for monitoring and future improvement.

Detailed definitions, formulas, measurement procedures, and quantitative thresholds for individual model and operational KPIs will be established separately in the KPI definition stage.

---

## 1.1.6 Success Measurement Approach

The success of the fraud detection solution will be evaluated using fraud-specific classification measures rather than relying only on overall accuracy.

The primary business considerations are:

- **Fraud detection effectiveness** — How successfully the system identifies actual fraudulent transactions.
- **Alert quality** — How effectively the system avoids incorrectly flagging legitimate transactions.
- **Overall detection balance** — Whether the system provides a practical balance between detecting fraud and limiting unnecessary alerts.

Metrics such as recall, precision, F1-score, PR-AUC, and other supporting measures will be used during model evaluation.

The detailed definition and calculation of these KPIs will be documented separately as part of **Phase 1.4 — KPI Definition**.

---

## 1.1.7 Accuracy Consideration

Overall accuracy will not be treated as the primary business success measure.

In a highly imbalanced fraud detection dataset, legitimate transactions may greatly outnumber fraudulent transactions. A model could therefore achieve a high overall accuracy while still failing to detect a significant proportion of fraudulent transactions.

For this reason, fraud-specific evaluation measures will receive greater importance during model development and evaluation.

The final choice of model and operating threshold should consider both:

- The ability to detect fraudulent transactions.
- The number of legitimate transactions incorrectly flagged as fraudulent.

---

## 1.1.8 Expected Business Outcome

The expected business outcome is an automated fraud detection capability that can assist financial institutions in identifying suspicious transactions more efficiently.

The intended business flow is:

```text
Large Volume of Transactions
            |
            v
Machine Learning-Based
Fraud Detection
            |
      +-----+------+
      |            |
      v            v
Potential       Legitimate
Fraud           Transactions
      |            |
      v            v
Flag for        Minimize
Investigation   Unnecessary Alerts
      |            |
      +-----+------+
            |
            v
Improved Fraud Detection Efficiency
            |
            v
Better Investigation Prioritization
            |
            v
Foundation for an
MLOps-Based Fraud Detection System