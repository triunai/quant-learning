import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from .base_signal import BaseSignal

class VolCompressionSignal(BaseSignal):
    def __init__(self):
        super().__init__("VolCompression")
        self.lookback = 20
        self.history_window = 252 # 1 year for percentile calculation
        self.rv_percentile = 0.0
        self.range_percentile = 0.0
        self.vix_percentile = 0.0

    def fit(self, data: pd.DataFrame):
        """
        In this context, 'fit' prepares the historical distributions.
        We assume 'data' contains enough history.
        """
        # Calculate metrics
        df = data.copy()

        # 1. Realized Volatility (20d)
        if 'Log_Ret' not in df.columns:
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        df['RV'] = df['Log_Ret'].rolling(self.lookback).std() * np.sqrt(252)

        # 2. Normalized Range (High-Low) / Close
        df['Norm_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Range_Smooth'] = df['Norm_Range'].rolling(self.lookback).mean()

        self.historical_data = df

    def predict(self, current_data: pd.DataFrame = None):
        """
        Calculate current compression score.
        If current_data is None, use the last row of historical data from fit().
        """
        if current_data is None:
            if not hasattr(self, 'historical_data'):
                raise ValueError("Signal must be fit first")
            df = self.historical_data
        else:
            # Append current data to history to update rolling metrics
            # For simplicity, let's assume current_data is the full dataframe including latest
            df = current_data

        # Ensure metrics are calculated
        if 'RV' not in df.columns:
            self.fit(df)
            df = self.historical_data

        # Get latest values
        latest = df.iloc[-1]

        # Get historical window for percentiles
        history = df.iloc[-self.history_window:]

        # 1. RV Percentile
        rv_rank = stats.percentileofscore(history['RV'].dropna(), latest['RV'])
        self.rv_percentile = rv_rank / 100.0

        # 2. Range Percentile
        range_rank = stats.percentileofscore(history['Range_Smooth'].dropna(), latest['Range_Smooth'])
        self.range_percentile = range_rank / 100.0

        # 3. VIX Percentile (Fetch VIX if not present)
        vix_score = 0.5 # Default neutral
        try:
            # We need to fetch VIX separately or assume it's passed?
            # Ideally passed, but for self-containment we can try fetching.
            # NOTE: This might slow down 'predict' if called frequently.
            # For this 'factory', let's assume we can fetch just the latest VIX.
            vix = yf.download("^VIX", period="1y", progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)

            if not vix.empty:
                vix_val = float(vix['Close'].iloc[-1])
                vix_rank = stats.percentileofscore(vix['Close'].values, vix_val)
                self.vix_percentile = float(vix_rank) / 100.0
            else:
                self.vix_percentile = 0.5
        except Exception as e:
            print(f"VIX fetch failed: {e}")
            self.vix_percentile = 0.5

        # COMBINE
        # Compression = Low RV, Low Range, Low VIX.
        # So we want 1 - percentile.

        score_components = [
            1.0 - self.rv_percentile,
            1.0 - self.range_percentile,
            1.0 - self.vix_percentile
        ]

        avg_compression = np.mean(score_components)

        # Output Score: 0-100
        self.confidence = avg_compression * 100.0

        # Logic:
        # If compressed (score > 70) -> Reduce size (storm coming), Direction?
        # Usually compression precedes expansion. Direction is unknown -> Neutral but defensive.
        # "Action: Reduce size, buy protection"

        self.reasoning = [
            f"RV Percentile: {self.rv_percentile:.2f}",
            f"Range Percentile: {self.range_percentile:.2f}",
            f"VIX Percentile: {self.vix_percentile:.2f}",
            f"Compression Score: {self.confidence:.1f}/100"
        ]

        if self.confidence > 70:
            self.direction = 0 # Neutral, but volatile
            self.position_sizing_mult = 0.5 # Reduce size
            self.reasoning.append("HIGH COMPRESSION: Reduce size, prepare for breakout.")
        elif self.confidence < 20:
             # High volatility already?
             self.direction = 0
             self.position_sizing_mult = 1.0 # Normal size?
             self.reasoning.append("Low compression (High Vol).")
        else:
             self.direction = 0
             self.position_sizing_mult = 1.0

        return {
            'score': self.confidence,
            'sizing': self.position_sizing_mult,
            'reasoning': self.reasoning
        }
