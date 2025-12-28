import numpy as np


def apply_narrative_tilt(self, sentiment_score):
    """
    Tilts the Markov matrix based on Jjules' sentiment score.
    """
    if abs(sentiment_score) < 0.1:
        return
    print(f"[JJULES] Tilting Matrix by {sentiment_score:+.2f}...")
    tilt_strength = 0.05 * abs(sentiment_score)
    matrix = self.markov_matrix.copy()
    for i in range(self.n_states):
        if sentiment_score > 0:
            matrix[i, 0] = max(0.01, matrix[i, 0] - tilt_strength)
            matrix[i, 4] += tilt_strength
        else:
            matrix[i, 4] = max(0.01, matrix[i, 4] - tilt_strength)
            matrix[i, 0] += tilt_strength
        matrix[i] = matrix[i] / np.sum(matrix[i])
    self.markov_matrix = matrix
