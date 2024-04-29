import logging
import logging.config
import os

import yaml

DEFAULT_CONFIG_FILE = 'config.yaml'


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(config_file_path=None):
    project_path = get_project_path()

    if config_file_path is None:
        config_file_path = DEFAULT_CONFIG_FILE

    config_path = os.path.join(project_path, config_file_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)

    return logger
