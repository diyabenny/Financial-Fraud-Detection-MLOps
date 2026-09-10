Phase 3 — Feast Feature Store Implementation Documentation

1. Objective

The objective of Phase 3 was to operationalize the validated Phase 3 feature layer using Feast while preserving the leakage-safe feature calculations developed by Avani.

The implementation separates responsibilities:

Python calculates the features → Feast manages, stores, and serves the feature values.

Feast is not responsible for calculating the rolling behavioral features.

2. Phase 3 Feature Set

The Phase 3 Feast FeatureView contains the following 10 model features.

Original transaction features

TX_AMOUNT

AMOUNT_LOG

TX_HOUR

TX_DAY_OF_WEEK

TX_IS_WEEKEND

Behavioral features

TERMINAL_TX_COUNT_1D

TERMINAL_TX_COUNT_7D

TERMINAL_AVG_AMOUNT_7D

CUSTOMER_TX_COUNT_1D

CUSTOMER_TX_COUNT_7D

The target variable TX_FRAUD is not included as a feature.

TX_FRAUD_SCENARIO is also excluded from the model feature set because it is evaluation-related information and must not be used as a predictive feature.

Feature approval: These 10 features should be treated as the approved final feature set only after Avani's final confirmation of their formulas, time windows, entity/time semantics, and leakage checks.

3. Feature Calculation and Leakage Prevention

The behavioral features are calculated before Feast using the validated Python feature-engineering logic.

For a transaction occurring at time T, only transactions strictly earlier than T are allowed to contribute.

1-day window

T - 1 day <= previous_time < T

7-day window

T - 7 days <= previous_time < T

Therefore:

The current transaction is excluded.

Same-timestamp transactions cannot influence each other.

Future transactions are excluded.

TX_FRAUD is not used.

TX_FRAUD_SCENARIO is not used.

May data is not used to construct April training/validation features.

This leakage-sensitive calculation remains outside Feast.

4. Feast Repository Structure

The production Feast repository was created at:

feast_repo/
└── fraud_features/
    ├── feature_repo/
    │   ├── feature_store.yaml
    │   ├── feature_definitions.py
    │   └── data/
    │
    ├── scripts/
    │   ├── may_online_processor.py
    │   ├── push_may_batches.py
    │   ├── verify_may_online.py
    │   └── compare_may_feast.py
    │
    └── tests/
        ├── test_feature_definitions.py
        ├── test_historical_retrieval.py
        ├── test_may_push.py
        └── test_online_push.py

The generated Feast registry, local online-store database, and Python cache files are ignored by Git.

5. Feast FileSource

The April training feature dataset was provided to Feast as a separate Parquet copy.

Source:

data/train_combined_features.parquet

It contains:

230,295 rows

18 columns

This is a Feast-specific copy of the validated Phase 3 training dataset. The original CSV was not modified.

The FileSource is defined as:

fraud_training_source = FileSource(
    name="fraud_training_source",
    path="data/train_combined_features.parquet",
    timestamp_field="TX_DATETIME",
)

Its purpose is to provide the historical/batch feature data for Feast.

6. Feast Entity

The transaction was defined as the Feast entity:

transaction = Entity(
    name="transaction",
    join_keys=["TRANSACTION_ID"],
)

Therefore:

Entity    = transaction
Join key  = TRANSACTION_ID

Each transaction is uniquely identified by TRANSACTION_ID.

The FeatureView is transaction-level: the 10 calculated feature values are attached to each transaction through TRANSACTION_ID. Although several features describe recent customer or terminal behavior, the stored feature record represents the feature values calculated for that specific transaction.

7. Feast PushSource

A PushSource was configured:

fraud_push_source = PushSource(
    name="fraud_push_source",
    batch_source=fraud_training_source,
)

The PushSource does not calculate features. Its role is to allow newly calculated feature values to be pushed into Feast's online serving layer.

Historical/development flow

April development data
        ↓
Python feature calculation
        ↓
FileSource
        ↓
FeatureView
        ↓
Historical retrieval / development validation

Later May final-test flow

May transaction
        ↓
Python feature calculation
        ↓
PushSource
        ↓
Feast Online Store
        ↓
Online retrieval

The April FileSource is the PushSource's batch source. This does not mean that Feast automatically pushes the April dataset into the Online Store.

8. Feast FeatureView

The FeatureView is:

fraud_features = FeatureView(
    name="fraud_features",
    entities=[transaction],
    ttl=None,
    schema=[
        Field(name="TX_AMOUNT", dtype=Float32),
        Field(name="AMOUNT_LOG", dtype=Float32),
        Field(name="TX_HOUR", dtype=Int64),
        Field(name="TX_DAY_OF_WEEK", dtype=Int64),
        Field(name="TX_IS_WEEKEND", dtype=Int64),
        Field(name="TERMINAL_TX_COUNT_1D", dtype=Float32),
        Field(name="TERMINAL_TX_COUNT_7D", dtype=Float32),
        Field(name="TERMINAL_AVG_AMOUNT_7D", dtype=Float32),
        Field(name="CUSTOMER_TX_COUNT_1D", dtype=Float32),
        Field(name="CUSTOMER_TX_COUNT_7D", dtype=Float32),
    ],
    online=True,
    source=fraud_push_source,
)

Feast confirmed:

FeatureView: fraud_features
Entity: transaction
Entity column: TRANSACTION_ID
Features: 10
Online: true
Batch source: fraud_training_source
Stream source: fraud_push_source

9. Feast Registration

The feature definitions were registered using:

feast apply

Result:

RESULT: PASS
Applying changes for project fraud_features
Updated feature view fraud_features
No changes to infrastructure

The only warning was:

Entity value_type will be mandatory in the next release.

This was a deprecation warning, not an error. The Feast configuration was successfully applied.

10. Historical Retrieval Test

A historical retrieval test was performed using Feast.

The retrieved feature values were compared with Avani's validated feature values.

Result:

RESULT: PASS
✓ Feast historical retrieval matches Avani's values.

This confirmed that Feast could retrieve the precomputed historical feature values correctly.

Important limitation: This test verifies that Feast retrieves the stored feature values correctly. It does not mean Feast calculated the behavioral features.

11. May Feature Processing

May contains:

182,429 transactions

May 1–May 19

TX_TIME_DAYS 30–48

The May processing script:

may_online_processor.py

calculates the leakage-safe behavioral features.

The May dataset was processed sequentially in 19 batches:

18 × 10,000 + 1 × 2,429 = 182,429 transactions

The resulting file was:

data/phase3/may_processed_features.csv

A separate Feast Parquet copy was also created:

data/may_processed_features.parquet

The original processed CSV was not modified for Feast.

May boundary clarification

During the initial Feast implementation, the May dataset was processed and pushed through the Feast workflow for technical validation. This happened before the May test-boundary correction was established.

That earlier run is retained as an implementation/technical validation result only. It is not used for training, feature selection, hyperparameter tuning, threshold selection, or champion model selection.

From the corrected workflow onward:

May
 ↓
LOCKED
 ↓
Final unseen evaluation only

12. May → Feast Online Store

May is reserved for the final unseen model evaluation.

The May online-ingestion workflow is retained for the later final-test stage, after the champion model has been frozen.

Champion model frozen
        ↓
May feature preparation
        ↓
Sequential May batch ingestion
        ↓
Feast Online Store
        ↓
Online retrieval
        ↓
Final prediction
        ↓
Final May evaluation

May-specific scripts retained for this stage:

may_online_processor.py

test_may_push.py

push_may_batches.py

verify_may_online.py

compare_may_feast.py

Current pre-champion Feast online validation uses non-test April/train development data.

13. Full May Batch Ingestion

Full May batch ingestion was implemented and technically tested during the initial Feast implementation, but it is reserved for the later final-test stage.

The May dataset must remain outside the current model-development and model-selection workflow.

The batch workflow uses:

push_may_batches.py

The May dataset contains:

182,429 transactions
19 batches
18 × 10,000 + 1 × 2,429

After the champion model is frozen, the intended workflow is:

Champion model frozen
        ↓
May feature preparation
        ↓
Sequential May batch ingestion
        ↓
Feast Online Store
        ↓
Online retrieval
        ↓
Final prediction
        ↓
Final May evaluation

The May scripts must not be executed again for model development or champion selection before the champion is frozen.

14. Online Retrieval Validation

Pre-champion online retrieval was validated using non-test development data.

The test:

test_online_push.py

uses:

feast_repo/fraud_features/feature_repo/data/train_combined_features.parquet

and selects the first row:

row = df.iloc[0]

The exact transaction used in the successful test was:

TRANSACTION_ID = 0
TX_DATETIME    = 2018-04-01 00:00:31
TX_TIME_DAYS   = 0

This is April train/development data and is independent of the May dataset.

The test pushes one known-good transaction containing all 10 registered features into Feast's Online Store and retrieves it successfully.

Result:

Push successful.

Online retrieval:
TRANSACTION_ID 0

This confirms:

Development feature values
        ↓
Feast PushSource
        ↓
Online Store
        ↓
Online Retrieval

The May-specific retrieval test, verify_may_online.py, is reserved for the later final-test stage.

15. Feast Feature-Value Consistency

Pre-champion feature-value consistency was validated using non-test development/training data.

Historical retrieval matched Avani's validated feature values:

RESULT: PASS
✓ Feast historical retrieval matches Avani's values.

The feature-definition consistency test also passed:

RESULT: PASS
✓ Feast feature definitions match the approved 10 features.

The test verifies that the Feast FeatureView contains exactly the 10 approved model features.

The May comparison script, compare_may_feast.py, is retained for the later final-test stage.

The previously completed May comparison is treated as a technical validation result only, not as pre-champion model evaluation.

16. Final Architecture

The architecture separates feature engineering, feature management, training retrieval, development online testing, and final testing.

Before champion selection

                AVANI
        Validated feature engineering
                    ↓
        Leakage-safe feature values
                    ↓
                  FEAST
                    ↓
               FeatureView
                /        \
               /          \
      Historical          Online
       Retrieval          Retrieval
          ↓                   ↓
   Training features    Development
                         online testing

Final May evaluation

May unseen test data
        ↓
Champion model frozen
        ↓
May feature preparation
        ↓
Feast Online Store
        ↓
Final prediction
        ↓
Final evaluation

Responsibility boundary

Avani

Calculates validated leakage-safe feature values

Diya / Feast

Registers, stores, retrieves, and serves those values

Swathi / Phase 4

Uses Feast retrieval for training and online prediction workflows

17. Important Architectural Decision

A key decision made during implementation was not to ask Feast to reconstruct the rolling behavioral features.

Feast's default historical point-in-time behavior uses timestamp matching that can include a feature row at the same timestamp. The project's behavioral feature logic requires:

previous_time < current_time

Therefore, the validated Python feature-engineering logic remains responsible for calculating the behavioral features.

Feast is responsible for:

Feature registration

Feature metadata

Historical feature retrieval

Online feature storage

Online feature retrieval

Serving the same feature layer to downstream ML workflows

This preserves the project's leakage-prevention rules.

18. Validation and Consistency Checks

A feature-definition consistency test verifies that the Feast FeatureView contains exactly the approved 10 model features.

The test compares feature names as sets, so Feast field ordering does not affect the result.

Result:

RESULT: PASS
✓ Feast feature definitions match the approved 10 features.

This complements the historical and online retrieval tests.

19. Challenges Encountered and Solutions

Challenge 1 — Feast's historical join behavior

Problem: Feast's default historical retrieval uses <= timestamp matching, while the project's requirement is strict <.

Solution: Behavioral features are precomputed using the validated Python logic rather than reconstructed by Feast.

Challenge 2 — Same-timestamp leakage

Problem: Same-time transactions could potentially be included by the default Feast historical join.

Solution: The behavioral feature calculations explicitly use strictly earlier transactions.

Challenge 3 — CSV FileSource

Problem: The Feast version being used does not provide CSV as the required FileSource format.

Solution: Separate Parquet copies were created for Feast.

Original Phase 3 CSV files remained unchanged.

Challenge 4 — May dataset missing four derived features

The processed May file contained the behavioral features but did not contain:

AMOUNT_LOG

TX_HOUR

TX_DAY_OF_WEEK

TX_IS_WEEKEND

Solution: These four deterministic features were calculated in the Feast push/verification scripts from the existing transaction amount and timestamp. The May source data itself was not modified.

Challenge 5 — Incorrect PushSource API usage

The initial test used:

store.push(source=...)

which caused an error.

The installed Feast 0.66 API expects:

store.push(
    push_source_name="fraud_push_source",
    df=push_df,
)

The script was corrected and the push succeeded.

Challenge 6 — False validation failure caused by data types

The first comparison reported:

FAIL: TX_HOUR
FAIL: TX_DAY_OF_WEEK

The actual values were not necessarily different; the test was requiring identical Pandas dtypes.

Solution: Numerical features were compared using:

np.allclose(...)

The final validation then passed.

Challenge 7 — May test-boundary correction

Problem: The initial Feast implementation processed and pushed the full May dataset before the final model-selection boundary was clarified.

Solution: The current online Feast demonstration was changed to non-test April/train development data. May-specific scripts are retained for the later final-test stage, and May is now locked against training, tuning, threshold selection, and champion selection until the champion model is frozen.

20. Final Results

Component

Result

Feast installation

✅

Feast repository

✅

Entity registration

✅

FileSource

✅

PushSource

✅

FeatureView

✅

10 features registered

✅

Feature-definition consistency

✅ PASS

Historical retrieval

✅ PASS

Development online push/retrieval

✅ PASS

May feature processing implementation

✅

May transactions technically processed

182,429

May batches

19

May full-batch push technical test

✅ PASS

May online retrieval technical test

✅ PASS

Feast vs processed May values technical test

✅ PASS

Original Phase 3 CSV modified

No

Leakage-safe behavioral calculation

✅ Preserved

May used for current pre-champion model development

No

May final evaluation

Deferred until champion model is frozen

The earlier May technical processing and validation are recorded as implementation results only. They are not used for current model training, tuning, threshold selection, or champion selection.

21. Phase 3 Completion Statement

The Phase 3 Feast implementation has been successfully implemented and technically validated.

The validated Python feature-engineering pipeline calculates the leakage-safe behavioral features, while Feast manages, stores, retrieves, and serves the feature values.

Pre-champion validation using non-test development data confirmed:

Historical retrieval matches Avani's validated feature values.

Online push and retrieval succeed.

The Feast FeatureView contains the approved 10 model features.

Feature-definition consistency validation passes.

The May processing and online-ingestion workflow was implemented and technically validated during the initial implementation. That earlier run is not used for model development or selection.

Under the corrected workflow, May remains locked and reserved for the final unseen evaluation after the champion model is frozen.

After the champion model is frozen, the retained May workflow can be used for final feature preparation, Feast ingestion/retrieval, prediction, and evaluation.

Phase 3 is ready for handoff to Phase 4 for model training, experiment tracking, model comparison, and model selection.