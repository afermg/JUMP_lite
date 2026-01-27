"""norm_2: Refactored morphological profile processing pipeline.

This package provides a simplified and streamlined implementation
of the norm/ pipeline, following trommel's clean architecture
with pycytominer/EFAAR-style APIs.

Key modules:
- core: Transformer classes and clean function APIs
- io: Data loading and saving utilities
- config: Dataclass-based configuration
- metrics: Combined phenotypic and batch metrics
- pipeline: Pipeline orchestration with Hydra integration

Example usage:

    from norm_2 import load_profiles, normalize, select_features, aggregate

    # Load data
    df = load_profiles("features.parquet")

    # Process
    df, stats = select_features(df, correlation_threshold_val=0.9)
    df = normalize(df, method="robustmad", batch_col="Metadata_Plate")
    df = aggregate(df, method="median")

    # Or use the full pipeline
    from norm_2.pipeline import run_pipeline
    run_pipeline(config_path="config.yaml")
"""

# Core transformers
from .core import (
    RobustMAD,
    Spherize,
    TVN,
    PCATransform,
)

# High-level API
from .core import (
    normalize,
    sample_normalize,
    select_features,
    aggregate,
    basic_cleanup,
)

# Feature selection functions
from .core import (
    variance_threshold,
    correlation_threshold,
    drop_na_columns,
    drop_outliers,
    blocklist_filter,
)

# I/O
from .io import (
    load_profiles,
    save_profiles,
    infer_columns,
    get_numeric_features,
    load_metadata,
)

# Configuration
from .config import (
    NormConfig,
    SelectConfig,
    StepConfig,
    PipelineConfig,
    create_default_config,
)

# Metrics
from .metrics import (
    calculate_phenotypic_activity,
    calculate_phenotypic_consistency,
    calculate_batch_metrics,
    evaluate_all,
)

# Pipeline
from .pipeline import run_pipeline, STEPS

__all__ = [
    # Transformers
    "RobustMAD",
    "Spherize",
    "TVN",
    "PCATransform",
    # High-level API
    "normalize",
    "sample_normalize",
    "select_features",
    "aggregate",
    "basic_cleanup",
    # Feature selection
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "drop_outliers",
    "blocklist_filter",
    # I/O
    "load_profiles",
    "save_profiles",
    "infer_columns",
    "get_numeric_features",
    "load_metadata",
    # Configuration
    "NormConfig",
    "SelectConfig",
    "StepConfig",
    "PipelineConfig",
    "create_default_config",
    # Metrics
    "calculate_phenotypic_activity",
    "calculate_phenotypic_consistency",
    "calculate_batch_metrics",
    "evaluate_all",
    # Pipeline
    "run_pipeline",
    "STEPS",
]
