"""
Orchestrate historical Feast feature retrieval
for training and validation datasets.
"""

from retrieve_training_features import main as retrieve_training_features
from retrieve_validation_features import main as retrieve_validation_features


def main():
    print("=" * 60)
    print("Starting Feast dataset preparation")
    print("=" * 60)

    print("\n[1/2] Retrieving training features from Feast...")
    retrieve_training_features()

    print("\n[2/2] Retrieving validation features from Feast...")
    retrieve_validation_features()

    print("\n" + "=" * 60)
    print("Feast dataset preparation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()