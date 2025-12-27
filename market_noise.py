import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from textblob import TextBlob
import warnings
import numpy as np

warnings.filterwarnings("ignore")

class JjulesNoiseMonitor:
    def __init__(self, ticker="PLTR"):
        self.ticker = ticker
        self.news_cache = []
        self.sentiment_score = 0
        self.narrative_regime = "NEUTRAL"

        # KEYWORDS (Weighted)
        self.keywords = {
            'bullish': {
                'contract': 2.0, 'award': 2.0, 'partnership': 1.5,
                'aip': 1.5, 'bootcamp': 1.0, 'expansion': 1.0,
                'beat': 2.0, 'profit': 1.5, 'sp500': 3.0,
                'inclusion': 2.0, 'buy': 1.0, 'upgrade': 1.5
            },
            'bearish': {
                'sell': 1.5, 'insider': 1.0, 'miss': 2.0,
                'loss': 1.5, 'downgrade': 1.5, 'lawsuit': 2.0,
                'delay': 1.5, 'overvalued': 1.0, 'dilution': 2.0,
                'soros': 0.5, 'karp': 0.5
            }
        }

    def fetch_finviz_news(self):
        print(f"[JJULES] Listening to the wire for {self.ticker}...")
        url = f"https://finviz.com/quote.ashx?t={self.ticker}&p=d"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            req = requests.get(url, headers=headers)
            soup = BeautifulSoup(req.content, 'html.parser')
            news_table = soup.find(id='news-table')
            parsed_news = []
            if not news_table:
                print("    [!] Could not find news table.")
                return
            for x in news_table.findAll('tr'):
                if not x.a:
                    continue
                text = x.a.get_text()
                date_scrape = x.td.text.split()
                if len(date_scrape) == 1:
                    time = date_scrape[0]
                    date = datetime.now().strftime('%Y-%m-%d')
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]
                if "loading" not in text.lower():
                    parsed_news.append([date, time, text])
            self.news_cache = parsed_news[:10]
            print(f"    Captured {len(self.news_cache)} recent headlines.")
        except Exception as e:
            print(f"    [!] Error fetching news: {e}")

    def analyze_sentiment(self):
        if not self.news_cache: return
        total_score = 0
        print("\n[JJULES] Decoding Narrative...")
        for news in self.news_cache:
            headline = news[2]
            headline_lower = headline.lower()
            blob_score = TextBlob(headline).sentiment.polarity
            kw_score = 0
            for word, weight in self.keywords['bullish'].items():
                if word in headline_lower: kw_score += weight
            for word, weight in self.keywords['bearish'].items():
                if word in headline_lower: kw_score -= weight
            total_score += blob_score + (kw_score * 0.5)
        self.sentiment_score = max(min(total_score / 3, 1.0), -1.0)
        if self.sentiment_score > 0.3: self.narrative_regime = "EUPHORIA"
        elif self.sentiment_score > 0.1: self.narrative_regime = "OPTIMISM"
        elif self.sentiment_score < -0.3: self.narrative_regime = "PANIC"
        elif self.sentiment_score < -0.1: self.narrative_regime = "FEAR"
        else: self.narrative_regime = "NOISE"

    def get_report_context(self):
        return f"[NARRATIVE] Score: {self.sentiment_score:+.2f} ({self.narrative_regime})"

if __name__ == "__main__":
    monitor = JjulesNoiseMonitor()
    monitor.fetch_finviz_news()
    monitor.analyze_sentiment()
    print(monitor.get_report_context())
