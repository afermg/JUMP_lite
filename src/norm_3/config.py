"""Configuration dataclasses for norm_3.

Provides type-safe configuration for GPU-accelerated pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class GPUConfig:
    """Configuration for GPU settings."""

    device: int = 0
    memory_limit_gb: float | None = None
    fallback_to_cpu: bool = True


@dataclass
class NormConfig:
    """Configuration for a normalization step."""

    method: Literal["robustmad", "standardize", "tvn", "spherize", "none"] = "robustmad"
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
        """Create from dictionary."""
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
    """Configuration for the full GPU-accelerated pipeline."""

    # Input/output
    input_path: str = ""
    output_path: str = "data/features/output.parquet"
    output_compression: str = "zstd"

    # GPU settings
    gpu: GPUConfig = field(default_factory=GPUConfig)

    # Steps (ordered list)
    steps: list[StepConfig] = field(default_factory=list)

    # Validation
    abort_on_invalid_features: bool = True
    max_pc1_variance: float = 0.50
    abort_on_ill_conditioned_tvn: bool = True

    @classmethod
    def from_hydra(cls, cfg: Any) -> "PipelineConfig":
        """Create PipelineConfig from Hydra DictConfig."""
        from omegaconf import OmegaConf

        if hasattr(cfg, '_content'):
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            config_dict = dict(cfg)

        # Extract input/output
        input_config = config_dict.get("input", {})
        output_config = config_dict.get("output", {})

        # Extract GPU config
        gpu_dict = config_dict.get("gpu", {})
        gpu_config = GPUConfig(
            device=gpu_dict.get("device", 0),
            memory_limit_gb=gpu_dict.get("memory_limit_gb"),
            fallback_to_cpu=gpu_dict.get("fallback_to_cpu", True),
        )

        # Parse steps
        steps = []
        for step_dict in config_dict.get("steps", []):
            steps.append(StepConfig.from_dict(step_dict))

        return cls(
            input_path=input_config.get("path", ""),
            output_path=output_config.get("path", "data/features/output.parquet"),
            output_compression=output_config.get("compression", "zstd"),
            gpu=gpu_config,
            steps=steps,
            abort_on_invalid_features=config_dict.get("abort_on_invalid_features", True),
            max_pc1_variance=config_dict.get("max_pc1_variance", 0.50),
            abort_on_ill_conditioned_tvn=config_dict.get("abort_on_ill_conditioned_tvn", True),
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


# Default step configurations for norm_3
DEFAULT_STEPS = [
    StepConfig(name="clean_nans", enabled=True, params={"na_cutoff": 0.30}),
    StepConfig(name="merge_metadata", enabled=True, params={}),
    StepConfig(
        name="normalize_robustmad",
        enabled=True,
        params={
            "batch_col": "Metadata_Plate",
            "fit_on_controls": True,
            "control_key": "negcon",
            "control_col": "Metadata_control_type",
        },
    ),
    StepConfig(name="normalize_spherize", enabled=False, params={"method": "ZCA-cor"}),
    StepConfig(name="normalize_tvn", enabled=False, params={"alpha": 0.5, "epsilon": 1.0}),
    StepConfig(name="normalize_pca", enabled=False, params={"n_components": 128}),
    StepConfig(
        name="aggregate_wells",
        enabled=True,
        params={"strata": ["Metadata_Plate", "Metadata_Well"], "method": "median"},
    ),
    StepConfig(name="evaluate_metrics", enabled=True, params={"skip_visualization": False}),
]


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return PipelineConfig(
        gpu=GPUConfig(),
        steps=[StepConfig(name=s.name, enabled=s.enabled, params=s.params.copy())
               for s in DEFAULT_STEPS],
    )
