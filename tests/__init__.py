import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.configuration.logger_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)
