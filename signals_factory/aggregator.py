from typing import List, Dict
from .base_signal import BaseSignal

class SignalAggregator:
    DEFAULT_WEIGHT = 0.1

    def __init__(self, signals: List[BaseSignal]):
        self.signals = signals
        # Weights could be dynamic based on backtest performance
        self.weights = {
            'VolCompression': 0.3,
            'RegimeFatigue': 0.25,
            'LiquidityHorizon': 0.2,
            'CrossAsset': 0.15,
            'EarningsRisk': 0.1
        }

    def aggregate(self) -> Dict:
        """
        Collect votes from all signals and compute the final 'Weather Report'.
        """
        combined_sizing = 1.0
        weighted_direction = 0.0
        total_weight = 0.0

        signal_reports = {}

        for signal in self.signals:
            # Assume predict() has been called or call it here if we pass data?
            # Ideally signals are already updated.
            # But let's assume we read the state from the signal objects.

            weight = self.weights.get(signal.name, self.DEFAULT_WEIGHT)

            # Sizing is multiplicative (conservative)
            combined_sizing *= signal.position_sizing_mult

            # Direction is weighted average
            if signal.direction != 0:
                weighted_direction += signal.direction * weight
                total_weight += weight

            signal_reports[signal.name] = {
                'direction': signal.direction,
                'confidence': signal.confidence,
                'sizing': signal.position_sizing_mult,
                'reasoning': signal.reasoning
            }

        final_direction = 0.0
        if total_weight > 0:
            final_direction = weighted_direction / total_weight

        # Round direction to -1, 0, 1 for simplicity in final report,
        # or keep as float for strength

        return {
            'final_sizing': combined_sizing,
            'final_score': final_direction, # -1 to 1
            'signals': signal_reports
        }
