import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from train_phase4 import (
    MODEL_FEATURES,
    TARGET,
    train_model,
    predict_probabilities,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/phase3/train_feast_features.csv"
EVALUATION_PATH = "data/phase4/april_25_30_evaluation_features.csv"
APRIL_SOURCE_PATH = "data/incoming/april/converted_april_file.csv"

OUTPUT_PATH = "reports/phase4/phase4_transaction_predictions.csv"


# ============================================================
# FROZEN CANDIDATES
# EXACTLY MATCHES evaluate_phase4_candidates.py
# ============================================================

CANDIDATES = [
    {
        "name": "LR_Original_C_0.01",
        "model_type": "logistic_regression",
        "imbalance_strategy": "original",
        "model_parameters": {
            "C": 0.01
        },
        "threshold": 0.10,
    },
    {
        "name": "RF_Balanced_100_None_1",
        "model_type": "random_forest",
        "imbalance_strategy": "balanced",
        "model_parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "threshold": 0.65,
    },
    {
        "name": "LR_Balanced_C_10",
        "model_type": "logistic_regression",
        "imbalance_strategy": "balanced",
        "model_parameters": {
            "C": 10
        },
        "threshold": 0.95,
    },
]


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_error_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "FPR": fpr,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4 - TRANSACTION LEVEL ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load frozen datasets
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    evaluation_df = pd.read_csv(
        EVALUATION_PATH
    )

    april_source = pd.read_csv(
        APRIL_SOURCE_PATH,
        usecols=[
            "TRANSACTION_ID",
            "TX_FRAUD_SCENARIO",
        ],
    )

    print(
        "\nTraining shape:",
        train_df.shape,
    )

    print(
        "Evaluation shape:",
        evaluation_df.shape,
    )

    print(
        "Training fraud:",
        train_df[TARGET].sum(),
    )

    print(
        "Evaluation fraud:",
        evaluation_df[TARGET].sum(),
    )

    # --------------------------------------------------------
    # Prepare model inputs
    # EXACT SAME 10 FEATURES
    # --------------------------------------------------------

    X_train = train_df[
        MODEL_FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_evaluation = evaluation_df[
        MODEL_FEATURES
    ]

    y_evaluation = evaluation_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Attach scenario information separately
    #
    # TX_FRAUD_SCENARIO is NOT used as a model feature.
    # It is retained only for later scenario analysis.
    # --------------------------------------------------------

    prediction_base = evaluation_df[
        [
            "TRANSACTION_ID",
            "TX_DATETIME",
            "TX_FRAUD",
        ]
    ].copy()

    prediction_base = prediction_base.merge(
        april_source,
        on="TRANSACTION_ID",
        how="left",
        validate="one_to_one",
    )

    if prediction_base[
        "TX_FRAUD_SCENARIO"
    ].isna().any():

        raise ValueError(
            "Some evaluation transactions are missing "
            "TX_FRAUD_SCENARIO."
        )

    print(
        "Scenario rows matched:",
        len(prediction_base),
    )

    # --------------------------------------------------------
    # Generate predictions for each frozen candidate
    # --------------------------------------------------------

    prediction_tables = []
    verification_results = []

    for candidate in CANDIDATES:

        print(
            "\n" + "-" * 70
        )

        print(
            "Candidate:",
            candidate["name"],
        )

        print(
            "Threshold:",
            candidate["threshold"],
        )

        # EXACT SAME TRAINING CALL
        # AS evaluate_phase4_candidates.py

        model = train_model(
            X_train,
            y_train,
            model_type=candidate[
                "model_type"
            ],
            imbalance_strategy=candidate[
                "imbalance_strategy"
            ],
            model_parameters=candidate[
                "model_parameters"
            ],
        )

        # EXACT SAME PROBABILITY GENERATION

        probabilities = predict_probabilities(
            model,
            X_evaluation,
        )

        # Apply the frozen threshold

        predicted_class = (
            probabilities
            >= candidate["threshold"]
        ).astype(int)

        # ----------------------------------------------------
        # Create transaction-level prediction table
        # ----------------------------------------------------

        result = prediction_base.copy()

        result["candidate"] = (
            candidate["name"]
        )

        result["predicted_probability"] = (
            probabilities
        )

        result["threshold"] = (
            candidate["threshold"]
        )

        result["predicted_class"] = (
            predicted_class
        )

        prediction_tables.append(
            result
        )

        # ----------------------------------------------------
        # Independent metric verification
        # ----------------------------------------------------

        metrics = calculate_error_metrics(
            y_evaluation,
            probabilities,
            candidate["threshold"],
        )

        metrics["candidate"] = (
            candidate["name"]
        )

        metrics["threshold"] = (
            candidate["threshold"]
        )

        verification_results.append(
            metrics
        )

        print(
            "TP:",
            metrics["TP"],
        )

        print(
            "FP:",
            metrics["FP"],
        )

        print(
            "TN:",
            metrics["TN"],
        )

        print(
            "FN:",
            metrics["FN"],
        )

        print(
            "Recall:",
            metrics["Recall"],
        )

        print(
            "Precision:",
            metrics["Precision"],
        )

        print(
            "F1:",
            metrics["F1"],
        )

        print(
            "FPR:",
            metrics["FPR"],
        )

    # --------------------------------------------------------
    # Combine prediction tables
    # --------------------------------------------------------

    predictions_df = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save prediction table
    # --------------------------------------------------------

    import os

    os.makedirs(
        "reports/phase4",
        exist_ok=True,
    )

    predictions_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Verification summary
    # --------------------------------------------------------

    verification_df = pd.DataFrame(
        verification_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "VERIFICATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        verification_df[
            [
                "candidate",
                "threshold",
                "TP",
                "FP",
                "TN",
                "FN",
                "Recall",
                "Precision",
                "F1",
                "FPR",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Expected totals
    # --------------------------------------------------------

    expected_fraud = 526

    expected_legitimate = (
        57767 - 526
    )

    print(
        "\nExpected fraud count:",
        expected_fraud,
    )

    print(
        "Expected legitimate count:",
        expected_legitimate,
    )

    # --------------------------------------------------------
    # Confusion matrix total verification
    # --------------------------------------------------------

    for _, row in verification_df.iterrows():

        fraud_total = (
            row["TP"]
            + row["FN"]
        )

        legitimate_total = (
            row["TN"]
            + row["FP"]
        )

        if fraud_total != expected_fraud:

            raise AssertionError(
                f"{row['candidate']}: "
                f"TP + FN = {fraud_total}, "
                f"expected {expected_fraud}"
            )

        if legitimate_total != expected_legitimate:

            raise AssertionError(
                f"{row['candidate']}: "
                f"TN + FP = {legitimate_total}, "
                f"expected {expected_legitimate}"
            )

    print(
        "\nCHECKPOINT PASSED"
    )

    print(
        "TP + FN = 526 for every candidate"
    )

    print(
        "TN + FP = 57241 for every candidate"
    )

    print(
        "\nPrediction file saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()