"""
Base classes for environment scenarios.
"""
from abc import ABC, abstractmethod

class Scenario(ABC):
    """
    Abstract base class for environment scenarios.
    Applies patches to the environment to modify request generation or step logic.
    """
    @abstractmethod
    def apply(self, env):
        pass
