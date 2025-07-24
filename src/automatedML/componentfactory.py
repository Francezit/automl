import logging

from .component import BaseComponent, BaseComponentSettings

#TODO
class ComponentFactory():
    _settings: BaseComponentSettings
    _logger: logging.Logger
    _seed: int

    def __init__(self,
                 settings: BaseComponentSettings,
                 logger: logging.Logger = None,
                 seed: int = None) -> None:
        self._settings = settings
        self._logger = logger
        self._seed = seed

