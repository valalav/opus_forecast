"""
Test suite for News Sentiment Analysis System
Tests CBR scraper, BERT classifier, and Sentiment Index generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class TestCBRScraper:
    """Test CBR Press Release Scraper"""

    def test_scraper_initialization(self):
        """Test that CBR scraper can be initialized"""
        from agents.news_sentiment import CBRScraper

        scraper = CBRScraper()
        assert scraper is not None
        assert hasattr(scraper, "base_url")
        assert hasattr(scraper, "scrape_press_releases")

    def test_scrape_recent_releases(self):
        """Test scraping of recent CBR press releases"""
        from agents.news_sentiment import CBRScraper

        scraper = CBRScraper()
        releases = scraper.scrape_press_releases(limit=5)
        assert isinstance(releases, list)
        assert len(releases) > 0
        assert all("date" in r for r in releases)
        assert all("title" in r for r in releases)
        assert all("content" in r for r in releases)

    def test_save_and_load_releases(self):
        """Test saving and loading scraped releases"""
        from agents.news_sentiment import CBRScraper
        import tempfile

        scraper = CBRScraper()
        test_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        test_path = test_file.name

        try:
            test_data = [
                {
                    "date": "2025-01-01",
                    "title": "Test Release",
                    "content": "Test content about interest rates and inflation.",
                }
            ]
            scraper.save_releases(test_data, test_path)
            loaded = scraper.load_releases(test_path)
            assert len(loaded) == 1
            assert loaded[0]["title"] == "Test Release"
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)


class TestHawkishnessClassifier:
    """Test BERT-based Hawkishness Classifier"""

    def test_classifier_initialization(self):
        """Test that classifier can be initialized"""
        from agents.news_sentiment import HawkishnessClassifier

        classifier = HawkishnessClassifier()
        assert classifier is not None
        assert hasattr(classifier, "predict")
        assert hasattr(classifier, "train")

    def test_predict_single_text(self):
        """Test prediction on single text"""
        from agents.news_sentiment import HawkishnessClassifier

        classifier = HawkishnessClassifier()

        hawkish_text = "The Board of Directors decided to raise the key rate to 16% to combat inflation."
        result = classifier.predict(hawkish_text)
        assert "label" in result
        assert "score" in result
        assert "confidence" in result
        assert result["label"] in ["hawkish", "dovish", "neutral"]

    def test_predict_batch(self):
        """Test batch prediction"""
        from agents.news_sentiment import HawkishnessClassifier

        classifier = HawkishnessClassifier()

        texts = [
            "We raised interest rates to fight inflation.",
            "Rates were lowered to stimulate growth.",
            "Monetary policy remains unchanged.",
        ]
        results = classifier.predict_batch(texts)
        assert len(results) == 3
        assert all("label" in r for r in results)

    def test_hawkishness_score_calculation(self):
        """Test hawkishness score (probability of hawkish stance)"""
        from agents.news_sentiment import HawkishnessClassifier

        classifier = HawkishnessClassifier()

        hawkish = "We will continue tightening monetary policy to control inflationary pressures."
        result = classifier.predict(hawkish)
        assert 0 <= result["hawkishness_score"] <= 1


class TestSentimentIndex:
    """Test Sentiment Index Generation"""

    def test_index_initialization(self):
        """Test that index can be initialized"""
        from agents.news_sentiment import SentimentIndex

        index = SentimentIndex()
        assert index is not None
        assert hasattr(index, "calculate_index")
        assert hasattr(index, "get_latest_score")

    def test_calculate_index_from_releases(self):
        """Test index calculation from press releases"""
        from agents.news_sentiment import SentimentIndex, HawkishnessClassifier

        test_releases = [
            {
                "date": "2025-01-15",
                "title": "Rate Hike",
                "content": "Key rate raised to 16%.",
            },
            {
                "date": "2025-01-20",
                "title": "Rate Cut",
                "content": "Key rate lowered to 15%.",
            },
            {
                "date": "2025-01-25",
                "title": "No Change",
                "content": "Key rate unchanged.",
            },
        ]

        index = SentimentIndex()
        df = index.calculate_index(test_releases)
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "hawkishness_score" in df.columns
        assert "label" in df.columns
        assert len(df) == 3

    def test_rolling_index_calculation(self):
        """Test rolling index calculation (smoothed over window)"""
        from agents.news_sentiment import SentimentIndex

        test_releases = [
            {
                "date": "2025-01-15",
                "title": "Rate Hike",
                "content": "Key rate raised to 16%.",
            },
            {
                "date": "2025-02-15",
                "title": "Rate Hike",
                "content": "Key rate raised to 17%.",
            },
            {
                "date": "2025-03-15",
                "title": "Rate Cut",
                "content": "Key rate lowered to 16%.",
            },
        ]

        index = SentimentIndex()
        df = index.calculate_index(test_releases, window=3)
        assert "rolling_index" in df.columns
        assert df["rolling_index"].notna().any()

    def test_export_index_to_csv(self):
        """Test exporting index to CSV"""
        from agents.news_sentiment import SentimentIndex
        import tempfile

        test_releases = [
            {"date": "2025-01-15", "title": "Test", "content": "Key rate decision."}
        ]

        index = SentimentIndex()
        test_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        test_path = test_file.name

        try:
            df = index.calculate_index(test_releases)
            index.export_to_csv(df, test_path)
            assert os.path.exists(test_path)

            loaded = pd.read_csv(test_path)
            assert len(loaded) == 1
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_pipeline(self):
        """Test end-to-end pipeline: scrape -> classify -> index"""
        from agents.news_sentiment import CBRScraper, SentimentIndex

        # 1. Scrape
        scraper = CBRScraper()
        releases = scraper.scrape_press_releases(limit=3)
        assert len(releases) > 0

        # 2. Classify and build index
        index = SentimentIndex()
        df = index.calculate_index(releases)
        assert len(df) > 0
        assert "hawkishness_score" in df.columns

        # 3. Export
        export_path = "test_sentiment_index.csv"
        try:
            index.export_to_csv(df, export_path)
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
