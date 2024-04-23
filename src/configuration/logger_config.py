import logging.config

import yaml


def setup_logging(config_file='../config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config['logger_config'])
