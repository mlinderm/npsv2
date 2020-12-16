from typing import NamedTuple

Config = NamedTuple

def merge_config(config: Config, updates) -> Config:
    # Adapted from snorkel: https://github.com/snorkel-team/snorkel/blob/master/snorkel/utils/config_utils.py
    for key, value in updates.items():
        if isinstance(value, dict):
            updates[key] = merge_config(getattr(config, key), value)
    return config._replace(**updates)