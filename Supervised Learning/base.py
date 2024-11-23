"""
This file stores the base class for Supervised Learning algorithms.
Note: The data file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library

# Third Party
from abc import ABC, abstractmethod
from pydantic import BaseModel


# Private


# -------------------------------------------------------------------------------------------------------------------- #

class SLBase(ABC):

    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("All subclasses must implement this method!")

    @abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError("All subclasses must implement this method!")

    @abstractmethod
    def scores(self, *args, **kwargs):
        raise NotImplementedError("All subclasses must implement this method!")