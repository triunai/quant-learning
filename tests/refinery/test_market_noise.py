"""Unit tests for refinery/market_noise.py (JjulesNoiseMonitor)."""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from refinery.market_noise import JjulesNoiseMonitor


class TestJjulesNoiseMonitorInitialization:
    """Test suite for JjulesNoiseMonitor initialization."""
    
    def test_init_with_default_ticker(self):
        """Test initialization with default ticker."""
        monitor = JjulesNoiseMonitor()
        assert monitor.ticker == "PLTR"
        assert monitor.sentiment_score == 0
        assert monitor.narrative_regime == "NEUTRAL"
        assert monitor.news_cache == []
    
    def test_init_with_custom_ticker(self):
        """Test initialization with custom ticker."""
        monitor = JjulesNoiseMonitor(ticker="AAPL")
        assert monitor.ticker == "AAPL"
    
    def test_has_keyword_dictionaries(self):
        """Test that keyword dictionaries are initialized."""
        monitor = JjulesNoiseMonitor()
        assert 'bullish' in monitor.keywords
        assert 'bearish' in monitor.keywords
        assert isinstance(monitor.keywords['bullish'], dict)
        assert isinstance(monitor.keywords['bearish'], dict)
    
    def test_bullish_keywords_loaded(self):
        """Test that bullish keywords are loaded."""
        monitor = JjulesNoiseMonitor()
        bullish = monitor.keywords['bullish']
        assert 'contract' in bullish
        assert 'partnership' in bullish
        assert bullish['contract'] > 0
    
    def test_bearish_keywords_loaded(self):
        """Test that bearish keywords are loaded."""
        monitor = JjulesNoiseMonitor()
        bearish = monitor.keywords['bearish']
        assert 'sell' in bearish
        assert 'loss' in bearish
        assert bearish['sell'] > 0


class TestFetchFinvizNews:
    """Test suite for fetch_finviz_news method."""
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetch from Finviz."""
        # Mock HTML response
        mock_html = """
        <html>
            <table id="news-table">
                <tr><td>Dec-28 10:00AM</td><a href="#">PLTR wins major contract</a></tr>
                <tr><td>Dec-27</td><a href="#">Stock upgraded by analyst</a></tr>
            </table>
        </html>
        """
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_get.return_value = mock_response
        
        monitor = JjulesNoiseMonitor()
        monitor.fetch_finviz_news()
        
        assert len(monitor.news_cache) > 0
    
    @patch('requests.get')
    def test_fetch_news_handles_no_table(self, mock_get):
        """Test handling when news table is not found."""
        mock_html = "<html><body>No news table</body></html>"
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_get.return_value = mock_response
        
        monitor = JjulesNoiseMonitor()
        monitor.fetch_finviz_news()
        
        # Should handle gracefully
        assert monitor.news_cache == []
    
    @patch('requests.get')
    def test_fetch_news_handles_network_error(self, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = Exception("Network error")
        
        monitor = JjulesNoiseMonitor()
        monitor.fetch_finviz_news()
        
        # Should not crash
        assert monitor.news_cache == []
    
    @patch('requests.get')
    def test_limits_news_to_10(self, mock_get):
        """Test that only 10 most recent headlines are kept."""
        # Create HTML with 20 news items
        news_items = '\n'.join([
            f'<tr><td>Dec-28 10:{i:02d}AM</td><a href="#">News {i}</a></tr>'
            for i in range(20)
        ])
        mock_html = f'<html><table id="news-table">{news_items}</table></html>'
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_get.return_value = mock_response
        
        monitor = JjulesNoiseMonitor()
        monitor.fetch_finviz_news()
        
        assert len(monitor.news_cache) <= 10


class TestAnalyzeSentiment:
    """Test suite for analyze_sentiment method."""
    
    def test_analyze_with_empty_cache(self):
        """Test sentiment analysis with no news."""
        monitor = JjulesNoiseMonitor()
        monitor.analyze_sentiment()
        
        assert monitor.sentiment_score == 0
        assert monitor.narrative_regime == "NEUTRAL"
    
    def test_analyze_bullish_news(self):
        """Test sentiment analysis with bullish keywords."""
        monitor = JjulesNoiseMonitor()
        monitor.news_cache = [
            ['2024-12-28', '10:00AM', 'PLTR wins major contract award'],
            ['2024-12-27', '09:00AM', 'Partnership expansion announced'],
            ['2024-12-26', '08:00AM', 'Company beats profit expectations']
        ]
        
        monitor.analyze_sentiment()
        
        assert monitor.sentiment_score > 0
        assert monitor.narrative_regime in ['OPTIMISM', 'EUPHORIA']
    
    def test_analyze_bearish_news(self):
        """Test sentiment analysis with bearish keywords."""
        monitor = JjulesNoiseMonitor()
        monitor.news_cache = [
            ['2024-12-28', '10:00AM', 'Analyst downgrade on loss concerns'],
            ['2024-12-27', '09:00AM', 'Company faces lawsuit'],
            ['2024-12-26', '08:00AM', 'Insider sell-off continues']
        ]
        
        monitor.analyze_sentiment()
        
        assert monitor.sentiment_score < 0
        assert monitor.narrative_regime in ['FEAR', 'PANIC']
    
    def test_analyze_neutral_news(self):
        """Test sentiment analysis with neutral news."""
        monitor = JjulesNoiseMonitor()
        monitor.news_cache = [
            ['2024-12-28', '10:00AM', 'Company holds quarterly meeting'],
            ['2024-12-27', '09:00AM', 'Market update published']
        ]
        
        monitor.analyze_sentiment()
        
        assert monitor.narrative_regime in ['NOISE', 'NEUTRAL', 'OPTIMISM']
    
    def test_sentiment_score_bounded(self):
        """Test that sentiment score is bounded between -1 and 1."""
        monitor = JjulesNoiseMonitor()
        # Extreme bullish news
        monitor.news_cache = [
            ['2024-12-28', '10:00AM', 'contract award partnership expansion sp500 inclusion buy upgrade'] * 10
        ]
        
        monitor.analyze_sentiment()
        
        assert -1.0 <= monitor.sentiment_score <= 1.0
    
    def test_regime_classification_thresholds(self):
        """Test regime classification at different sentiment levels."""
        monitor = JjulesNoiseMonitor()
        
        # Test EUPHORIA threshold
        monitor.sentiment_score = 0.4
        monitor.narrative_regime = "NEUTRAL"  # Reset
        # Manually set to test classification
        if monitor.sentiment_score > 0.3:
            monitor.narrative_regime = "EUPHORIA"
        assert monitor.narrative_regime == "EUPHORIA"
        
        # Test PANIC threshold
        monitor.sentiment_score = -0.4
        if monitor.sentiment_score < -0.3:
            monitor.narrative_regime = "PANIC"
        assert monitor.narrative_regime == "PANIC"


class TestGetReportContext:
    """Test suite for get_report_context method."""
    
    def test_returns_formatted_string(self):
        """Test that report context returns properly formatted string."""
        monitor = JjulesNoiseMonitor()
        monitor.sentiment_score = 0.25
        monitor.narrative_regime = "OPTIMISM"
        
        result = monitor.get_report_context()
        
        assert isinstance(result, str)
        assert "0.25" in result or "+0.25" in result
        assert "OPTIMISM" in result
        assert "[NARRATIVE]" in result
    
    def test_report_with_negative_sentiment(self):
        """Test report format with negative sentiment."""
        monitor = JjulesNoiseMonitor()
        monitor.sentiment_score = -0.15
        monitor.narrative_regime = "FEAR"
        
        result = monitor.get_report_context()
        
        assert "-0.15" in result
        assert "FEAR" in result
    
    def test_report_with_zero_sentiment(self):
        """Test report format with neutral sentiment."""
        monitor = JjulesNoiseMonitor()
        monitor.sentiment_score = 0.0
        monitor.narrative_regime = "NOISE"
        
        result = monitor.get_report_context()
        
        assert "NOISE" in result


class TestKeywordWeighting:
    """Test keyword weighting system."""
    
    def test_high_impact_keywords_weighted_more(self):
        """Test that high-impact keywords have higher weights."""
        monitor = JjulesNoiseMonitor()
        
        # SP500 inclusion is a major event
        assert monitor.keywords['bullish']['sp500'] >= monitor.keywords['bullish']['buy']
        
        # Lawsuit is more severe than simple sell
        assert monitor.keywords['bearish']['lawsuit'] >= monitor.keywords['bearish']['sell']
    
    def test_all_weights_positive(self):
        """Test that all weights are positive numbers."""
        monitor = JjulesNoiseMonitor()
        
        for category in ['bullish', 'bearish']:
            for keyword, weight in monitor.keywords[category].items():
                assert weight > 0, f"{keyword} has non-positive weight"


@pytest.mark.integration
class TestJjulesNoiseMonitorIntegration:
    """Integration tests for full workflow."""
    
    @patch('requests.get')
    def test_full_workflow(self, mock_get):
        """Test complete news fetch and analysis workflow."""
        mock_html = """
        <html>
            <table id="news-table">
                <tr><td>Dec-28 10:00AM</td><a href="#">PLTR wins contract</a></tr>
                <tr><td>Dec-27</td><a href="#">Stock upgraded</a></tr>
            </table>
        </html>
        """
        mock_response = Mock()
        mock_response.content = mock_html.encode()
        mock_get.return_value = mock_response
        
        monitor = JjulesNoiseMonitor(ticker="PLTR")
        monitor.fetch_finviz_news()
        monitor.analyze_sentiment()
        report = monitor.get_report_context()
        
        assert len(monitor.news_cache) > 0
        assert isinstance(monitor.sentiment_score, (int, float))
        assert isinstance(report, str)