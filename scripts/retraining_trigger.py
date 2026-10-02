"""
Step 16C - Retraining trigger.

Uses the existing Step 16B drift result to decide whether
candidate retraining should be requested.

This script does NOT train, evaluate, or replace the Champion.
"""

from detect_feature_drift import (
    add_deterministic_features,
    calculate_drift,
    load_populations,
    validate_features,
)


def get_retraining_decision():
    """
    Determine whether candidate retraining should be requested.

    Returns:
        dict: Retraining decision and supporting drift information.
    """
    reference_df, current_df = load_populations()

    current_df = add_deterministic_features(current_df)

    validate_features(
        reference_df,
        current_df,
    )

    _, overall_drift_detected, crossing_features = calculate_drift(
        reference_df,
        current_df,
    )

    if overall_drift_detected:
        return {
            "overall_drift_detected": True,
            "retraining_requested": True,
            "action": "REQUEST_STEP_17_RETRAINING",
            "crossing_features": crossing_features,
        }

    return {
        "overall_drift_detected": False,
        "retraining_requested": False,
        "action": "KEEP_RF300_CHAMPION",
        "crossing_features": [],
    }


def main():
    """Run the Step 16C retraining decision."""
    decision = get_retraining_decision()

    print("=" * 70)
    print("STEP 16C - RETRAINING TRIGGER")
    print("=" * 70)

    print(
        "Overall drift detected:",
        decision["overall_drift_detected"],
    )

    print(
        "Retraining requested:",
        decision["retraining_requested"],
    )

    print(
        "Action:",
        decision["action"],
    )

    print(
        "Crossing features:",
        decision["crossing_features"],
    )


if __name__ == "__main__":
    main()
