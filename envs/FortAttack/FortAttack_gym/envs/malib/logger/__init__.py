"""Logger module.

This module instantiates a global logger singleton.
"""
from FortAttack_gym.envs.malib.logger.histogram import Histogram
from FortAttack_gym.envs.malib.logger.logger import Logger, LogOutput
from FortAttack_gym.envs.malib.logger.simple_outputs import StdOutput, TextOutput
from FortAttack_gym.envs.malib.logger.tabular_input import TabularInput
from FortAttack_gym.envs.malib.logger.csv_output import CsvOutput  # noqa: I100
from FortAttack_gym.envs.malib.logger.snapshotter import Snapshotter
from FortAttack_gym.envs.malib.logger.tensor_board_output import TensorBoardOutput

logger = Logger()
tabular = TabularInput()
snapshotter = Snapshotter()

__all__ = [
    'Histogram', 'Logger', 'CsvOutput', 'StdOutput', 'TextOutput', 'LogOutput',
    'Snapshotter', 'TabularInput', 'TensorBoardOutput', 'logger', 'tabular',
    'snapshotter'
]
