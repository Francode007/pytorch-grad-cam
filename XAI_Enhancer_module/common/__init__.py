"""Shared utilities for the CMPB revision (splits, training matrix, stats)."""

from XAI_Enhancer_module.common.splits import (
    GroupIdError,
    deduplicate_across_splits,
    diagnose_ibs_grouping,
    extract_group_id,
    load_group_map,
    prepare_ibs_patient_folds,
    prepare_kvasir_splits,
    write_split_summary,
)

__all__ = [
    "GroupIdError",
    "deduplicate_across_splits",
    "diagnose_ibs_grouping",
    "extract_group_id",
    "load_group_map",
    "prepare_ibs_patient_folds",
    "prepare_kvasir_splits",
    "write_split_summary",
]
