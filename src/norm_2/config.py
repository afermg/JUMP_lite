"""Configuration dataclasses for morphological profile processing.

Provides type-safe configuration for:
- Normalization methods
- Feature selection
- Pipeline orchestration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NormConfig:
    """Configuration for a normalization step."""

    method: Literal["robustmad", "standardize", "robustize", "tvn", "spherize", "none"] = "robustmad"
    batch_col: str | None = "Metadata_Plate"
    fit_on_controls: bool = False
    control_col: str = "Metadata_control_type"
    control_key: str = "negcon"

    # TVN-specific
    tvn_alpha: float = 0.5
    tvn_epsilon: float = 1.0

    # Spherize-specific
    spherize_method: str = "ZCA-cor"
    spherize_epsilon: float = 1e-6

    # RobustMAD-specific
    epsilon: float = 1e-18

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormConfig":
        """Create from dictionary (e.g., Hydra config)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SelectConfig:
    """Configuration for feature selection."""

    variance_threshold: float | None = None
    freq_cut: float = 0.05
    unique_cut: float = 0.01
    outlier_cutoff: float = 500
    correlation_threshold: float = 0.9
    na_cutoff: float = 0.30

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelectConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StepConfig:
    """Configuration for a pipeline step."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StepConfig":
        """Create from dictionary."""
        return cls(
            name=d.get("name", ""),
            enabled=d.get("enabled", True),
            params=d.get("params", {}),
        )


@dataclass
class PipelineConfig:
    """Configuration for the full normalization pipeline."""

    # Input/output
    input_path: str = ""
    output_path: str = "data/features/output.parquet"
    output_compression: str = "zstd"

    # Steps (ordered list)
    steps: list[StepConfig] = field(default_factory=list)

    # Validation
    abort_on_invalid_features: bool = True
    max_pc1_variance: float = 0.50
    validate_batch_correction: bool = False

    # Sweep options
    skip_redundant_configs: bool = False

    @classmethod
    def from_hydra(cls, cfg: Any) -> "PipelineConfig":
        """
        Create PipelineConfig from Hydra DictConfig.

        This handles the Hydra-specific structure with nested steps.
        """
        from omegaconf import OmegaConf

        if hasattr(cfg, '_content'):
            # DictConfig
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            config_dict = dict(cfg)

        # Extract input/output
        input_config = config_dict.get("input", {})
        output_config = config_dict.get("output", {})

        # Parse steps
        steps = []
        for step_dict in config_dict.get("steps", []):
            steps.append(StepConfig.from_dict(step_dict))

        return cls(
            input_path=input_config.get("path", ""),
            output_path=output_config.get("path", "data/features/output.parquet"),
            output_compression=output_config.get("compression", "zstd"),
            steps=steps,
            abort_on_invalid_features=config_dict.get("abort_on_invalid_features", True),
            max_pc1_variance=config_dict.get("max_pc1_variance", 0.50),
            validate_batch_correction=config_dict.get("validate_batch_correction", False),
            skip_redundant_configs=config_dict.get("skip_redundant_configs", False),
        )

    def get_step(self, name: str) -> StepConfig | None:
        """Get step configuration by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def is_step_enabled(self, name: str) -> bool:
        """Check if a step is enabled."""
        step = self.get_step(name)
        return step.enabled if step else False


# Default step configurations
DEFAULT_STEPS = [
    StepConfig(name="clean_nans", enabled=True, params={"na_cutoff": 0.30}),
    StepConfig(name="sample_norm", enabled=False, params={"norm": "l2"}),
    StepConfig(name="merge_metadata", enabled=True, params={"metadata_dir": "analysis/feature_similarity/input"}),
    StepConfig(
        name="filter_features",
        enabled=False,
        params={
            "operations": [
                {"name": "variance_threshold", "freq_cut": 0.05, "unique_cut": 0.01},
                {"name": "drop_outliers", "outlier_cutoff": 500},
            ]
        },
    ),
    StepConfig(
        name="prune_correlated",
        enabled=True,
        params={"threshold": 0.9, "method": "pearson"},
    ),
    StepConfig(
        name="normalize_standard",
        enabled=True,
        params={
            "method": "robustmad",
            "batch_col": "Metadata_Plate",
            "fit_on_controls": False,
            "control_key": "negcon",
            "control_col": "Metadata_control_type",
        },
    ),
    StepConfig(name="normalize_tvn", enabled=False, params={"alpha": 0.5, "epsilon": 1.0}),
    StepConfig(
        name="normalize_spherize",
        enabled=False,
        params={"method": "ZCA-cor", "epsilon": 1e-6},
    ),
    StepConfig(name="normalize_harmony", enabled=False, params={}),
    StepConfig(name="normalize_combat", enabled=False, params={}),
    StepConfig(
        name="aggregate_wells",
        enabled=True,
        params={"strata": ["Metadata_Plate", "Metadata_Well"], "method": "median"},
    ),
    StepConfig(name="evaluate_metrics", enabled=True, params={"skip_visualization": False}),
]


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return PipelineConfig(steps=DEFAULT_STEPS.copy())
