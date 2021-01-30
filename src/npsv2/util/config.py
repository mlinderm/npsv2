from typing import NamedTuple

Config = NamedTuple

def merge_config(config: Config, updates) -> Config:
    # Adapted from snorkel: https://github.com/snorkel-team/snorkel/blob/master/snorkel/utils/config_utils.py
    filtered_updates = {}
    for key, value in updates.items():
        if key not in config:
            continue
        if isinstance(value, dict):
            filtered_updates[key] = merge_config(getattr(config, key), value)
        else:
            filtered_updates[key] = value
    return config._replace(**filtered_updates)