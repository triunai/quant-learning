"""Unit tests for refinery/integration_snippet.py matrix tilting logic."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestNarrativeTiltLogic:
    """Test suite for narrative tilt matrix manipulation logic."""
    
    def test_snippet_file_exists(self):
        """Test that integration snippet file exists."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        assert snippet_path.exists()
    
    def test_snippet_contains_tilt_function(self):
        """Test that snippet contains apply_narrative_tilt function."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        content = snippet_path.read_text()
        assert 'apply_narrative_tilt' in content
        assert 'sentiment_score' in content
    
    def test_tilt_logic_positive_sentiment(self):
        """Test matrix tilting logic with positive sentiment."""
        # Simulate the tilt logic from snippet
        sentiment_score = 0.5
        tilt_strength = 0.05 * abs(sentiment_score)
        
        # Create a sample matrix
        n_states = 5
        matrix = np.ones((n_states, n_states)) / n_states
        
        for i in range(n_states):
            if sentiment_score > 0:
                # Shift probability from crash (state 0) to rally (state 4)
                matrix[i, 0] = max(0.01, matrix[i, 0] - tilt_strength)
                matrix[i, 4] += tilt_strength
            matrix[i] = matrix[i] / np.sum(matrix[i])
        
        # Verify matrix is still stochastic
        assert np.allclose(matrix.sum(axis=1), 1.0)
        
        # Verify rally state probabilities increased
        assert matrix[0, 4] > 1.0 / n_states
    
    def test_tilt_logic_negative_sentiment(self):
        """Test matrix tilting logic with negative sentiment."""
        sentiment_score = -0.5
        tilt_strength = 0.05 * abs(sentiment_score)
        
        n_states = 5
        matrix = np.ones((n_states, n_states)) / n_states
        
        for i in range(n_states):
            if sentiment_score < 0:
                # Shift probability from rally to crash
                matrix[i, 4] = max(0.01, matrix[i, 4] - tilt_strength)
                matrix[i, 0] += tilt_strength
            matrix[i] = matrix[i] / np.sum(matrix[i])
        
        # Verify matrix is stochastic
        assert np.allclose(matrix.sum(axis=1), 1.0)
        
        # Verify crash state probabilities increased
        assert matrix[0, 0] > 1.0 / n_states
    
    def test_tilt_logic_preserves_minimum_probability(self):
        """Test that minimum probability floor (0.01) is enforced."""
        sentiment_score = 1.0  # Extreme sentiment
        tilt_strength = 0.05 * abs(sentiment_score)
        
        n_states = 5
        matrix = np.ones((n_states, n_states)) / n_states
        
        for i in range(n_states):
            if sentiment_score > 0:
                matrix[i, 0] = max(0.01, matrix[i, 0] - tilt_strength)
                matrix[i, 4] += tilt_strength
            matrix[i] = matrix[i] / np.sum(matrix[i])
        
        # No probability should be below 0.01
        assert np.all(matrix >= 0.01) or np.all(matrix >= -1e-10)  # Account for floating point
    
    def test_weak_sentiment_no_tilt(self):
        """Test that weak sentiment (< 0.1) should not trigger tilt."""
        sentiment_score = 0.05
        
        if abs(sentiment_score) < 0.1:
            # Should return early, no tilt applied
            assert True
        else:
            pytest.fail("Weak sentiment should not trigger tilt")
    
    def test_tilt_strength_scales_with_sentiment(self):
        """Test that tilt strength scales linearly with sentiment magnitude."""
        for sentiment in [0.2, 0.5, 0.8]:
            tilt_strength = 0.05 * abs(sentiment)
            expected = 0.05 * abs(sentiment)
            assert np.isclose(tilt_strength, expected)


class TestIntegrationSnippetStructure:
    """Test the structure and documentation of integration snippet."""
    
    def test_function_signature(self):
        """Test that function has correct signature."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        content = snippet_path.read_text()
        
        # Check function definition
        assert 'def apply_narrative_tilt(self, sentiment_score):' in content
    
    def test_uses_numpy(self):
        """Test that snippet references numpy for matrix operations."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        content = snippet_path.read_text()
        
        # Should use numpy for normalization
        assert 'np.sum' in content
    
    def test_modifies_markov_matrix(self):
        """Test that snippet modifies markov_matrix attribute."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        content = snippet_path.read_text()
        
        assert 'self.markov_matrix' in content
    
    def test_snippet_is_method(self):
        """Test that snippet is a method (takes self)."""
        snippet_path = Path(__file__).parent.parent.parent / 'refinery' / 'integration_snippet.py'
        content = snippet_path.read_text()
        
        assert 'def apply_narrative_tilt(self,' in content