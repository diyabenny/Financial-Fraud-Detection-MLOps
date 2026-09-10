import pandas as pd
from bisect import bisect_left
from collections import defaultdict


# ============================================================
# PROJECT DATA PATHS
# ============================================================

APRIL_TRAIN_PATH = r"data\phase3\train_combined_features.csv"
APRIL_VALIDATION_PATH = r"data\phase3\validation_combined_features.csv"
MAY_SOURCE_PATH = r"data\incoming\may\converted_may_file.csv"


# ============================================================
# OUTPUT PATH
# ============================================================

OUTPUT_PATH = r"data\phase3\may_processed_features.csv"


# ============================================================
# MAY BATCH CONFIGURATION
# ============================================================

BATCH_SIZE = 10000


# ============================================================
# TIME CONSTANTS
# ============================================================

ONE_DAY_NS = 24 * 60 * 60 * 1_000_000_000
SEVEN_DAYS_NS = 7 * ONE_DAY_NS


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

BEHAVIORAL_FEATURES = [
    "TERMINAL_TX_COUNT_1D",
    "TERMINAL_TX_COUNT_7D",
    "TERMINAL_AVG_AMOUNT_7D",
    "CUSTOMER_TX_COUNT_1D",
    "CUSTOMER_TX_COUNT_7D",
]


# ============================================================
# LOAD APRIL HISTORY
# ============================================================

def load_april_history():
    """Load all April transactions as the initial historical state."""

    train_df = pd.read_csv(
        APRIL_TRAIN_PATH
    )

    validation_df = pd.read_csv(
        APRIL_VALIDATION_PATH
    )

    april_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True
    )

    april_df["TX_DATETIME"] = pd.to_datetime(
        april_df["TX_DATETIME"],
        errors="raise"
    )

    april_df = april_df.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    return april_df


# ============================================================
# LOAD MAY DATA
# ============================================================

def load_may_data():
    """Load May transactions in chronological order."""

    may_df = pd.read_csv(
        MAY_SOURCE_PATH
    )

    may_df["TX_DATETIME"] = pd.to_datetime(
        may_df["TX_DATETIME"],
        errors="raise"
    )

    may_df = may_df.sort_values(
        "TX_DATETIME"
    ).reset_index(drop=True)

    return may_df


# ============================================================
# CALCULATE FEATURES FOR ONE TRANSACTION
# ============================================================

def calculate_features_for_transaction(
    transaction,
    history
):
    """
    Calculate leakage-safe behavioral features
    for one transaction.

    This function is kept as a simple reference/
    validation implementation.
    """

    current_time = transaction["TX_DATETIME"]

    start_1d = current_time - pd.Timedelta(days=1)

    start_7d = current_time - pd.Timedelta(days=7)

    # Strictly earlier transactions only.
    previous_history = history[
        history["TX_DATETIME"] < current_time
    ]

    # --------------------------------------------------------
    # CUSTOMER HISTORY
    # --------------------------------------------------------

    customer_history = previous_history[
        previous_history["CUSTOMER_ID"]
        == transaction["CUSTOMER_ID"]
    ]

    customer_1d = customer_history[
        customer_history["TX_DATETIME"] >= start_1d
    ]

    customer_7d = customer_history[
        customer_history["TX_DATETIME"] >= start_7d
    ]

    # --------------------------------------------------------
    # TERMINAL HISTORY
    # --------------------------------------------------------

    terminal_history = previous_history[
        previous_history["TERMINAL_ID"]
        == transaction["TERMINAL_ID"]
    ]

    terminal_1d = terminal_history[
        terminal_history["TX_DATETIME"] >= start_1d
    ]

    terminal_7d = terminal_history[
        terminal_history["TX_DATETIME"] >= start_7d
    ]

    # --------------------------------------------------------
    # TERMINAL AVERAGE
    # --------------------------------------------------------

    terminal_avg_7d = (
        terminal_7d["TX_AMOUNT"].mean()
        if len(terminal_7d) > 0
        else 0.0
    )

    return {
        "TERMINAL_TX_COUNT_1D": float(
            len(terminal_1d)
        ),
        "TERMINAL_TX_COUNT_7D": float(
            len(terminal_7d)
        ),
        "TERMINAL_AVG_AMOUNT_7D": float(
            terminal_avg_7d
        ),
        "CUSTOMER_TX_COUNT_1D": float(
            len(customer_1d)
        ),
        "CUSTOMER_TX_COUNT_7D": float(
            len(customer_7d)
        ),
    }


# ============================================================
# PROCESS MAY BATCHES
# ============================================================

def process_may_batches(
    april_history,
    may_df
):
    """
    Process May transactions sequentially in batches.

    Leakage rule:
        previous_time < current_time

    Therefore:
        - current transaction is excluded
        - same-time transactions are excluded
        - future transactions are excluded
    """

    # --------------------------------------------------------
    # HISTORY INDEXES
    # --------------------------------------------------------

    customer_times = defaultdict(list)

    terminal_times = defaultdict(list)

    terminal_amount_prefix = defaultdict(list)

    # --------------------------------------------------------
    # BUILD INDEX FROM APRIL
    # --------------------------------------------------------

    april_columns = [
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_DATETIME",
        "TX_AMOUNT",
    ]

    for (
        customer_id,
        terminal_id,
        tx_datetime,
        tx_amount,
    ) in april_history[
        april_columns
    ].itertuples(
        index=False,
        name=None
    ):

        timestamp_ns = int(
            tx_datetime.value
        )

        customer_id = int(
            customer_id
        )

        terminal_id = int(
            terminal_id
        )

        tx_amount = float(
            tx_amount
        )

        # Customer timestamp history
        customer_times[
            customer_id
        ].append(
            timestamp_ns
        )

        # Terminal timestamp history
        terminal_times[
            terminal_id
        ].append(
            timestamp_ns
        )

        # Terminal cumulative amount
        prefix = terminal_amount_prefix[
            terminal_id
        ]

        if not prefix:
            prefix.append(0.0)

        prefix.append(
            prefix[-1] + tx_amount
        )

    # --------------------------------------------------------
    # PROCESS MAY BATCHES
    # --------------------------------------------------------

    processed_batches = []

    for batch_number, start in enumerate(
        range(
            0,
            len(may_df),
            BATCH_SIZE
        ),
        start=1
    ):

        batch = may_df.iloc[
            start:start + BATCH_SIZE
        ].copy()

        batch_results = []

        # ----------------------------------------------------
        # PROCESS EACH TRANSACTION
        # ----------------------------------------------------

        for transaction in batch.itertuples(
            index=False
        ):

            customer_id = int(
                transaction.CUSTOMER_ID
            )

            terminal_id = int(
                transaction.TERMINAL_ID
            )

            current_ns = int(
                transaction.TX_DATETIME.value
            )

            start_1d_ns = (
                current_ns - ONE_DAY_NS
            )

            start_7d_ns = (
                current_ns - SEVEN_DAYS_NS
            )

            # ------------------------------------------------
            # CUSTOMER FEATURES
            # ------------------------------------------------

            customer_history = customer_times[
                customer_id
            ]

            customer_1d_start = bisect_left(
                customer_history,
                start_1d_ns
            )

            customer_1d_end = bisect_left(
                customer_history,
                current_ns
            )

            customer_7d_start = bisect_left(
                customer_history,
                start_7d_ns
            )

            customer_7d_end = bisect_left(
                customer_history,
                current_ns
            )

            customer_tx_count_1d = (
                customer_1d_end
                - customer_1d_start
            )

            customer_tx_count_7d = (
                customer_7d_end
                - customer_7d_start
            )

            # ------------------------------------------------
            # TERMINAL FEATURES
            # ------------------------------------------------

            terminal_history = terminal_times[
                terminal_id
            ]

            terminal_1d_start = bisect_left(
                terminal_history,
                start_1d_ns
            )

            terminal_1d_end = bisect_left(
                terminal_history,
                current_ns
            )

            terminal_7d_start = bisect_left(
                terminal_history,
                start_7d_ns
            )

            terminal_7d_end = bisect_left(
                terminal_history,
                current_ns
            )

            terminal_tx_count_1d = (
                terminal_1d_end
                - terminal_1d_start
            )

            terminal_tx_count_7d = (
                terminal_7d_end
                - terminal_7d_start
            )

            # ------------------------------------------------
            # TERMINAL 7-DAY AVERAGE
            # ------------------------------------------------

            terminal_prefix = (
                terminal_amount_prefix[
                    terminal_id
                ]
            )

            terminal_sum_7d = (
                terminal_prefix[
                    terminal_7d_end
                ]
                - terminal_prefix[
                    terminal_7d_start
                ]
            )

            if terminal_tx_count_7d > 0:

                terminal_avg_amount_7d = (
                    terminal_sum_7d
                    / terminal_tx_count_7d
                )

            else:

                terminal_avg_amount_7d = 0.0

            # ------------------------------------------------
            # CREATE RESULT
            # ------------------------------------------------

            result = transaction._asdict()

            result.update({
                "TERMINAL_TX_COUNT_1D":
                    float(
                        terminal_tx_count_1d
                    ),

                "TERMINAL_TX_COUNT_7D":
                    float(
                        terminal_tx_count_7d
                    ),

                "TERMINAL_AVG_AMOUNT_7D":
                    float(
                        terminal_avg_amount_7d
                    ),

                "CUSTOMER_TX_COUNT_1D":
                    float(
                        customer_tx_count_1d
                    ),

                "CUSTOMER_TX_COUNT_7D":
                    float(
                        customer_tx_count_7d
                    ),
            })

            batch_results.append(
                result
            )

            # ------------------------------------------------
            # ADD CURRENT TRANSACTION TO HISTORY
            # ------------------------------------------------

            customer_times[
                customer_id
            ].append(
                current_ns
            )

            terminal_times[
                terminal_id
            ].append(
                current_ns
            )

            terminal_prefix = (
                terminal_amount_prefix[
                    terminal_id
                ]
            )

            if not terminal_prefix:
                terminal_prefix.append(0.0)

            terminal_prefix.append(
                terminal_prefix[-1]
                + float(
                    transaction.TX_AMOUNT
                )
            )

        # ----------------------------------------------------
        # CREATE PROCESSED BATCH
        # ----------------------------------------------------

        processed_batch = pd.DataFrame(
            batch_results
        )

        processed_batches.append(
            processed_batch
        )

        print(
            f"Batch {batch_number}: "
            f"{len(processed_batch)} "
            f"transactions processed."
        )

    # --------------------------------------------------------
    # COMBINE ALL PROCESSED BATCHES
    # --------------------------------------------------------

    return pd.concat(
        processed_batches,
        ignore_index=True
    )


# ============================================================
# MAIN - FULL MAY PROCESSING
# ============================================================

if __name__ == "__main__":

    # Load April as initial historical state
    april_history = load_april_history()

    # Load ALL May transactions
    # Expected: 182,429 transactions
    # Expected: 19 batches
    may_df = load_may_data()

    # Process all May transactions sequentially
    result = process_may_batches(
        april_history,
        may_df
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print(
        "\nProcessed May transactions:"
    )

    print(
        result[
            ["TRANSACTION_ID"]
            + BEHAVIORAL_FEATURES
        ].head(10).to_string(
            index=False
        )
    )

    print(
        f"\nTotal transactions processed: "
        f"{len(result)}"
    )

    # --------------------------------------------------------
    # SAVE PROCESSED MAY FEATURES
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved processed May features to: "
        f"{OUTPUT_PATH}"
    )