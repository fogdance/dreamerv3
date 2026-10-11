"""Canonical YAML reader shared by the CLI and experiment tools."""
from pathlib import Path

from ruamel.yaml import YAML


def read_config_yaml(path):
  """Preserve YAML types, including scientific floats and quoted strings."""
  data = YAML(typ='safe').load(Path(path).read_text())
  if not isinstance(data, dict):
    raise ValueError(f'Dreamer configuration must be a mapping: {path}')
  return data
