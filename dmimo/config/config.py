"""Abstract class for configuration.
"""

from abc import ABC
import copy


class Config(ABC):
    """Abstract configuration class.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key in dir(self):
                setattr(self, key, value)

    def _ifndef(self, name, value):
        if not hasattr(self, f"_{name}"):
            setattr(self, f"_{name}", value)

    def clone(self, deepcopy=True):
        """Returns a copy of the Config object
        """
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def verify(self):
        pass
