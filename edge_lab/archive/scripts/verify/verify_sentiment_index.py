#!/usr/bin/env python3
"""
Verification script for Task 101: Cognitive Architecture - News Sentiment

Acceptance Criteria: "Sentiment Index created"

This script verifies:
1. News sentiment module exists and is importable
2. Sentiment index can be generated from CBR releases
3. Index contains required columns (date, hawkishness_score, label, rolling_index)
4. Index is exported to CSV file
5. Summary statistics are available
"""

import os
import sys
import pandas as pd


def verify_module_exists():
    """Verify news_sentiment module exists"""
    print("✓ Checking module existence...")
    try:
        from agents.news_sentiment import (
            CBRScraper,
            HawkishnessClassifier,
            SentimentIndex,
            run_full_pipeline,
        )

        print("  ✓ Module imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        return False


def verify_scraper():
    """Verify CBR scraper works"""
    print("\n✓ Checking CBR scraper...")
    try:
        from agents.news_sentiment import CBRScraper

        scraper = CBRScraper()
        releases = scraper.scrape_press_releases(limit=3)

        if len(releases) == 0:
            print("  ✗ No releases scraped")
            return False

        if not all("date" in r and "title" in r and "content" in r for r in releases):
            print("  ✗ Release format incorrect")
            return False

        print(f"  ✓ Scraped {len(releases)} releases")
        return True
    except Exception as e:
        print(f"  ✗ Scraper failed: {e}")
        return False


def verify_classifier():
    """Verify hawkishness classifier works"""
    print("\n✓ Checking Hawkishness classifier...")
    try:
        from agents.news_sentiment import HawkishnessClassifier

        classifier = HawkishnessClassifier()

        hawkish = "The Board of Directors decided to raise the key rate to 16% to combat inflation."
        dovish = "The key rate was lowered to stimulate economic growth."
        neutral = "Monetary policy remains unchanged."

        h_result = classifier.predict(hawkish)
        d_result = classifier.predict(dovish)
        n_result = classifier.predict(neutral)

        if h_result["label"] != "hawkish":
            print(f"  ✗ Hawkish classification failed: got {h_result['label']}")
            return False

        if d_result["label"] != "dovish":
            print(f"  ✗ Dovish classification failed: got {d_result['label']}")
            return False

        if n_result["label"] != "neutral":
            print(f"  ✗ Neutral classification failed: got {n_result['label']}")
            return False

        print("  ✓ Classifier correctly identifies hawkish/dovish/neutral")
        return True
    except Exception as e:
        print(f"  ✗ Classifier failed: {e}")
        return False


def verify_sentiment_index():
    """Verify sentiment index generation"""
    print("\n✓ Checking Sentiment Index generation...")
    try:
        from agents.news_sentiment import CBRScraper, SentimentIndex

        scraper = CBRScraper()
        releases = scraper.scrape_press_releases(limit=5)

        index = SentimentIndex()
        df = index.calculate_index(releases)

        required_columns = ["date", "hawkishness_score", "label", "rolling_index"]
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            print(f"  ✗ Missing columns: {missing}")
            return False

        print(f"  ✓ Index generated with {len(df)} rows")
        print(f"  ✓ Columns: {list(df.columns)}")

        # Check score range
        if df["hawkishness_score"].min() < 0 or df["hawkishness_score"].max() > 1:
            print(f"  ✗ Hawkishness scores out of range [0, 1]")
            return False

        print("  ✓ Scores in valid range [0, 1]")
        return True
    except Exception as e:
        print(f"  ✗ Index generation failed: {e}")
        return False


def verify_csv_export():
    """Verify CSV export"""
    print("\n✓ Checking CSV export...")
    try:
        from agents.news_sentiment import run_full_pipeline

        csv_path = "data/sentiment_index.csv"
        df = run_full_pipeline(csv_path, limit=3)

        if not os.path.exists(csv_path):
            print(f"  ✗ CSV file not created at {csv_path}")
            return False

        loaded_df = pd.read_csv(csv_path)

        if len(loaded_df) != len(df):
            print(f"  ✗ CSV row count mismatch: {len(loaded_df)} vs {len(df)}")
            return False

        print(f"  ✓ CSV created with {len(loaded_df)} rows")
        print(f"  ✓ Path: {csv_path}")
        return True
    except Exception as e:
        print(f"  ✗ CSV export failed: {e}")
        return False


def verify_summary():
    """Verify summary statistics"""
    print("\n✓ Checking summary statistics...")
    try:
        from agents.news_sentiment import CBRScraper, SentimentIndex

        scraper = CBRScraper()
        releases = scraper.scrape_press_releases(limit=5)

        index = SentimentIndex()
        df = index.calculate_index(releases)
        summary = index.get_historical_summary(df)

        required_keys = [
            "period_start",
            "period_end",
            "mean_hawkishness",
            "total_releases",
            "hawkish_count",
            "dovish_count",
            "neutral_count",
        ]

        if not all(key in summary for key in required_keys):
            missing = set(required_keys) - set(summary.keys())
            print(f"  ✗ Missing summary keys: {missing}")
            return False

        print("  ✓ Summary statistics available")
        print(f"    Period: {summary['period_start']} to {summary['period_end']}")
        print(f"    Mean hawkishness: {summary['mean_hawkishness']:.3f}")
        print(
            f"    Releases: {summary['total_releases']} (H:{summary['hawkish_count']}, D:{summary['dovish_count']}, N:{summary['neutral_count']})"
        )
        return True
    except Exception as e:
        print(f"  ✗ Summary generation failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 70)
    print("Task 101 Verification: News Sentiment Index")
    print("=" * 70)

    checks = [
        verify_module_exists,
        verify_scraper,
        verify_classifier,
        verify_sentiment_index,
        verify_csv_export,
        verify_summary,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("  ✓ ALL CHECKS PASSED")
        print("  ✓ Acceptance Criteria met: 'Sentiment Index created'")
        print("=" * 70)
        return 0
    else:
        print("  ✗ SOME CHECKS FAILED")
        print("  ✗ Acceptance Criteria NOT met")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
