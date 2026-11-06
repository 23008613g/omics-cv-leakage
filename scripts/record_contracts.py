"""Unchanged formal record types/counts extracted from the original assembly script."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RecordContract:
    relative_source: str
    rows: int
    fields: int
    scope: str

CANONICAL_RECORDS = {'dataset_eligibility_flow.csv': RecordContract('formal_qc_v4/dataset_eligibility_flow.csv', 1102, 25, 'one row per attempted dataset'), 'run_inventory.csv': RecordContract('formal_tidy_v4/run_inventory.csv', 1102, 33, 'one row per attempted dataset'), 'human_cohort_adjudication_v2_manual.csv': RecordContract('metadata/human_cohort_adjudication_v2_manual.csv', 1102, 41, 'one row per attempted dataset'), 'dataset_method_summary.csv': RecordContract('formal_tidy_v4/dataset_method_summary.csv', 11957, 21, 'one row per dataset, method and available arm'), 'dataset_method_uncertainty.csv': RecordContract('formal_tidy_v4/dataset_method_uncertainty.csv', 5435, 21, 'one row per observed-complete dataset and supervised method'), 'permutation_summary.csv': RecordContract('formal_tidy_v4/permutation_summary.csv', 32610, 26, 'one row per dataset, method and permutation metric'), 'feature_stability.csv': RecordContract('formal_tidy_v4/feature_stability.csv', 5435, 26, 'one row per observed-complete dataset and supervised method'), 'outer_fold_predictions.csv': RecordContract('formal_tidy_v4/outer_fold_predictions.csv', 3327060, 16, 'one row per dataset, method, arm, repeat, fold and test observation'), 'outer_fold_diagnostics.csv': RecordContract('formal_tidy_v4/outer_fold_diagnostics.csv', 178695, 28, 'one row per observed outer fit'), 'failures.csv': RecordContract('formal_tidy_v4/failures.csv', 45, 12, 'one row per recorded failure event'), 'meta_input.csv': RecordContract('meta_input_v4/meta_input.csv', 5430, 50, 'one row per analysis-eligible dataset and supervised method'), 'subset_audit.csv': RecordContract('meta_analysis_v4/subset_audit.csv', 1086, 26, 'one row per analysis-eligible dataset')}
