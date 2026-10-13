from dataclasses import dataclass
import yaml


@dataclass
class Settings:
    """All tunable numbers of the pipeline live here.

    Default values are defined below. If a config.yaml file exists
    in the project root, its values override the defaults.
    """
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    missing_ratio_limit: float = 0.3
    classification_threshold: float = 0.75
    regression_threshold: float = 0.6

    max_rows_for_kernel_models: int = 10000
    enable_tuning: bool = True
    tuning_trials: int = 30

    column_missing_limit: float = 0.4   # drop columns with more missing than this

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Settings":
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}  # no yaml file -> just use defaults

        # safety: ignore any key in yaml that doesn't match a real setting
        # (protects against typos like "test_szie: 0.3" crashing the app)
        valid_keys = cls.__dataclass_fields__.keys()
        clean_data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**clean_data)


# one shared instance — every agent imports this same object
settings = Settings.from_yaml()