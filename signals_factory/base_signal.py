from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt

class BaseSignal(ABC):
    def __init__(self, name: str):
        self.name = name
        self.direction = 0  # -1 (Short), 0 (Neutral), 1 (Long)
        self.confidence = 0.0  # 0 to 100
        self.position_sizing_mult = 1.0  # 0 to 1
        self.time_horizon = 1  # days
        self.reasoning = []

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        Train the signal model on historical data.
        """
        pass

    @abstractmethod
    def predict(self, current_data: pd.DataFrame):
        """
        Generate a signal for the current state.
        Updates self.direction, self.confidence, etc.
        Returns a dict with signal details.
        """
        pass

    def backtest(self, data: pd.DataFrame):
        """
        Run a walk-forward backtest.
        Default implementation can be overridden.
        """
        pass

    def plot(self, data: pd.DataFrame):
        """
        Visualize the signal.
        """
        pass
