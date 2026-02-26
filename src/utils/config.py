import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require(cfg: dict, keypath: str):
    """
    keypath: "train.lr" のようにドット区切り
    """
    cur = cfg
    for k in keypath.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {keypath}")
        cur = cur[k]
    return cur


def optional(cfg: dict, keypath: str, default=None):
    cur = cfg
    for k in keypath.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
