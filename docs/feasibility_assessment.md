## 1.2 ML Task Definition

The business problem is to identify fraudulent financial transactions while reducing unnecessary alerts for legitimate transactions.

To address this business problem using machine learning, the task is formulated as a **supervised binary classification problem**. The model will learn from historical transaction data with known fraud outcomes and use transaction information available at the time of prediction to classify a transaction as legitimate or fraudulent.

The prediction target is `TX_FRAUD`, where:

- `0` represents a legitimate transaction.
- `1` represents a fraudulent transaction.

Class imbalance is treated as a data and modeling challenge rather than the ML problem itself. Similarly, changing fraud patterns over time are considered a system and MLOps challenge that will be addressed during later stages of the project.

### 1.2.1 Problem Type

The financial fraud detection task is a **supervised binary classification problem**.

Historical transaction records are available together with their corresponding fraud labels. This allows the model to learn patterns from previously labelled transactions and apply those patterns when making predictions on previously unseen transactions.

The model will classify each transaction into one of two classes:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

Therefore, the ML problem is to classify a transaction as legitimate or fraudulent based on the transaction information available at the time of prediction.

### 1.2.2 Target Variable

The target variable selected for the fraud detection task is `TX_FRAUD`.

It is a binary variable representing the fraud status of each transaction:

| Target Value | Meaning |
|---:|---|
| 0 | Legitimate transaction |
| 1 | Fraudulent transaction |

The dataset contains **288,236 transactions** in total. Of these, **286,512 transactions are legitimate** and **1,724 transactions are fraudulent**.

Fraudulent transactions therefore represent approximately **0.60%** of the dataset, while legitimate transactions represent approximately **99.40%**.

This distribution indicates a severe class imbalance. The imbalance is considered a data and modeling challenge that will need to be addressed during later stages of model development and evaluation.

### 1.2.3 Input Features

The dataset contains information related to transactions, customers, terminals, and transaction timing.

The main variables that can potentially provide predictive information include:

- `TX_DATETIME` — records the date and time of the transaction and can be transformed into useful time-based features during preprocessing.
- `TX_AMOUNT` — represents the transaction amount.
- `TX_TIME_SECONDS` — represents the transaction time in seconds.
- `TX_TIME_DAYS` — represents the transaction time in days.

The dataset also contains `CUSTOMER_ID` and `TERMINAL_ID`. These variables identify the customer and terminal associated with each transaction. Since their numerical values represent identifiers rather than continuous measurements, they should not automatically be treated as ordinary numerical features. They may be considered later for appropriate encoding or for deriving meaningful customer- or terminal-level behavioral features.

`TRANSACTION_ID` is unique for each transaction and is used to identify individual records. It is therefore not considered a meaningful predictive feature.

`TX_FRAUD` is the target variable and will not be used as an input feature.

`TX_FRAUD_SCENARIO` requires special consideration. During the initial dataset inspection, we observed that `TX_FRAUD_SCENARIO = 0` occurs for legitimate transactions, while non-zero values occur for fraudulent transactions. Using this variable directly to predict `TX_FRAUD` could therefore expose information about the target and introduce target leakage. For this reason, it will be excluded from the initial predictive feature set unless a leakage-free use is established during later analysis.

### 1.2.4 Prediction Objective

The model's prediction objective is to **predict whether each transaction belongs to the legitimate or fraudulent class**, using only information available at the time of prediction.

The resulting predictions can be used to identify transactions that may require further attention. The business can then use these predictions to prioritize potentially suspicious transactions and manage fraud alerts.

Since fraudulent transactions represent only a small proportion of the dataset, overall accuracy alone will not be sufficient to assess model performance. Fraud-sensitive evaluation measures will therefore be considered during later model development and evaluation.

Changing fraud patterns over time are also an important consideration for the overall system. However, monitoring and responding to such changes will be addressed during the later MLOps stages rather than being part of the definition of the ML classification task.

## 1.3 Feasibility Assessment

The feasibility of the proposed fraud detection task was assessed using the available financial transaction dataset.

The assessment considers the size and structure of the dataset, available variables, data types, missing values, duplicate records, class distribution, and potential data quality concerns.

The initial assessment indicates that the dataset provides a suitable basis for developing the proposed supervised binary classification task. However, several factors, including severe class imbalance, treatment of identifier variables, timestamp processing, and potential target leakage, need to be considered during the subsequent stages of the project.

Overall, the feasibility assessment is intended to determine whether the available data is suitable for proceeding with the planned machine learning workflow and to identify important data-related considerations before model development.

### 1.3.1 Dataset Availability

A historical financial transaction dataset is available for developing the proposed fraud detection model.

The dataset contains **288,236 transaction records** and **9 columns**. The records include transaction information, customer and terminal identifiers, transaction timing information, the fraud indicator, and a fraud scenario field.

The availability of historical transaction records together with a clearly defined fraud target makes the dataset suitable for the proposed supervised learning task.

### 1.3.2 Dataset Characteristics

The dataset contains **288,236 transaction records** across **9 columns**.

The available variables are:

- `TRANSACTION_ID`
- `TX_DATETIME`
- `CUSTOMER_ID`
- `TERMINAL_ID`
- `TX_AMOUNT`
- `TX_TIME_SECONDS`
- `TX_TIME_DAYS`
- `TX_FRAUD`
- `TX_FRAUD_SCENARIO`

The dataset contains **4,958 unique customers** and **10,000 unique terminals**.

There are also **288,236 unique transaction IDs**, which is equal to the total number of records. This indicates that each transaction has a distinct transaction identifier.

The presence of transaction, customer, terminal, amount, and time-related information provides a useful basis for further data analysis and feature development.

### 1.3.3 Data Types

The initial inspection shows that most variables are stored as numerical values, while `TX_DATETIME` is currently stored as a string.

The observed data types are:

| Column | Data Type |
|---|---|
| `TRANSACTION_ID` | Integer |
| `TX_DATETIME` | String |
| `CUSTOMER_ID` | Integer |
| `TERMINAL_ID` | Integer |
| `TX_AMOUNT` | Float |
| `TX_TIME_SECONDS` | Integer |
| `TX_TIME_DAYS` | Integer |
| `TX_FRAUD` | Integer |
| `TX_FRAUD_SCENARIO` | Integer |

`TX_DATETIME` will require appropriate datetime processing before time-based information can be extracted from it.

Although `CUSTOMER_ID` and `TERMINAL_ID` are stored as integers, they represent identifiers rather than continuous numerical measurements. They will therefore require appropriate treatment during feature engineering.

### 1.3.4 Missing Values

The dataset was checked for missing values across all nine columns.

No missing values were found in any of the columns.

This is a positive aspect of the initial data quality assessment because the current dataset does not require missing-value treatment based on the initial inspection. However, missing-value checks should still be included as part of the data preprocessing workflow to ensure that future data is handled consistently.

### 1.3.5 Duplicate Records

The dataset was checked for completely duplicated rows.

The inspection found **0 duplicate records**.

This indicates that there are no exact duplicate rows in the current dataset. Additional consistency checks may still be performed during preprocessing if required.

### 1.3.6 Class Distribution

The target variable `TX_FRAUD` is highly imbalanced.

The observed class distribution is:

| Class | Meaning | Count | Percentage |
|---:|---|---:|---:|
| 0 | Legitimate | 286,512 | 99.40% |
| 1 | Fraudulent | 1,724 | 0.60% |
| **Total** | | **288,236** | **100%** |

Only approximately **0.60%** of the transactions are fraudulent, while approximately **99.40%** are legitimate.

This represents a significant class imbalance and is an important data and modeling challenge for the project. A model that predicts most transactions as legitimate could achieve high overall accuracy while still failing to identify a significant number of fraudulent transactions.

Therefore, appropriate class-imbalance handling and fraud-sensitive evaluation measures will need to be considered during later model development and evaluation.

### 1.3.7 Data Quality

The initial data quality assessment shows several positive characteristics. The dataset contains no missing values and no completely duplicated records, and the target variable is clearly defined.

However, some variables require additional processing before model training.

`TX_DATETIME` is currently stored as a string and will need to be converted into an appropriate datetime format for further analysis and feature extraction.

`CUSTOMER_ID` and `TERMINAL_ID` are identifiers rather than ordinary numerical measurements and will require careful treatment during feature engineering.

`TX_FRAUD_SCENARIO` also requires special consideration. During the initial inspection, we observed that `TX_FRAUD_SCENARIO = 0` occurs for legitimate transactions, while non-zero values occur for fraudulent transactions. Using this variable directly to predict `TX_FRAUD` could therefore introduce target leakage. It should be excluded from the initial predictive feature set unless a leakage-free use is established later.

The severe class imbalance is another important data-related challenge that will need to be considered during model development and evaluation.

### 1.3.8 Technical Feasibility

The available dataset is technically suitable for developing the proposed supervised fraud classification task.

The dataset contains a substantial number of historical transaction records with a clearly defined fraud target. Transaction amount and time-related information provide a starting point for feature development, while customer and terminal information may support additional behavioral features after appropriate processing.

The main technical challenges identified are class imbalance, treatment of identifier variables, timestamp processing, and prevention of target leakage.

These challenges can be addressed during the subsequent preprocessing, feature engineering, model development, and evaluation stages. Based on the initial assessment, no major data-related limitation was identified that would prevent the proposed ML task from being developed.

### 1.3.9 Interpretability

Interpretability is an important consideration for a fraud detection system because transactions identified as potentially fraudulent may require further investigation.

The available transaction information, such as transaction amount and timing, provides features that can be related to transaction behavior.

However, some derived features and machine learning models may make individual predictions more difficult to interpret. Therefore, model interpretation and feature importance should be considered during later stages of model development and evaluation.

The final system should provide useful information about the factors contributing to fraud predictions where possible.

### 1.3.10 Feasibility Conclusion

Based on the initial assessment, the available dataset is **feasible for developing a supervised machine-learning-based financial fraud detection system**.

The feasibility is supported by the availability of **288,236 historical transaction records**, relevant transaction, timing, customer, and terminal information, and a clearly defined binary fraud target. Importantly, the dataset contains **1,724 known fraudulent transactions**, providing a substantial number of labelled fraudulent transactions for the planned fraud classification task.

No missing values or completely duplicated records were identified during the initial inspection.

The available transaction, timing, customer, and terminal information provides a suitable basis for further analysis and feature development.

The main challenges identified are the severe class imbalance, careful treatment of identifier variables, timestamp processing, and potential target leakage associated with `TX_FRAUD_SCENARIO`.

These challenges do not prevent the project from proceeding, but they will need to be addressed during subsequent stages of preprocessing, feature engineering, model development, and evaluation.

Overall, the feasibility assessment supports proceeding with the proposed fraud detection ML task.