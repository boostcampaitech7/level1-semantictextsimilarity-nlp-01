import yaml

def load_config(config_path='./config.yaml'):
    """
    Parameters:
    - config_path: str, default='./config.yaml'
        Path to the config file.
    Returns:
    - config: dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config