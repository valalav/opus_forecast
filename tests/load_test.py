#!/usr/bin/env python3
"""
Load Testing Script for СИРЕНА API.

Simple load test using Python threading to simulate concurrent users.
Measures latency, throughput, and identifies performance degradation.

Usage:
    python3 tests/load_test.py --endpoint /health --users 10 --requests 100
    python3 tests/load_test.py --endpoint /forecast/quick --users 10 --requests 50
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ ERROR: 'requests' library not installed. Run: pip install requests")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class LoadTestResult:
    """Result of a single request."""

    success: bool
    status_code: int
    latency_ms: float
    error_message: str = ""


@dataclass
class LoadTestSummary:
    """Summary statistics of load test."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    total_duration_sec: float
    latency_degradation_pct: float = 0.0


class LoadTester:
    """Load testing engine using threading."""

    def __init__(self, base_url: str, endpoint: str):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint.lstrip("/")
        self.full_url = f"{self.base_url}/{self.endpoint}"
        self.results: List[LoadTestResult] = []
        self.lock = threading.Lock()

    def make_request(self) -> LoadTestResult:
        """Make a single request and measure latency."""
        try:
            start_time = time.perf_counter()
            response = requests.get(self.full_url, timeout=10)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            return LoadTestResult(
                success=response.status_code == 200,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
        except requests.exceptions.Timeout:
            return LoadTestResult(
                success=False,
                status_code=0,
                latency_ms=10000.0,  # Timeout value
                error_message="Request timeout",
            )
        except requests.exceptions.ConnectionError:
            return LoadTestResult(
                success=False,
                status_code=0,
                latency_ms=0.0,
                error_message="Connection error",
            )
        except Exception as e:
            return LoadTestResult(
                success=False,
                status_code=0,
                latency_ms=0.0,
                error_message=str(e),
            )

    def run_worker(self, num_requests: int):
        """Worker thread that makes multiple requests."""
        worker_results = []
        for _ in range(num_requests):
            result = self.make_request()
            worker_results.append(result)

        # Thread-safe append
        with self.lock:
            self.results.extend(worker_results)

    def run_load_test(
        self,
        num_users: int = 10,
        requests_per_user: int = 10,
    ) -> LoadTestSummary:
        """
        Run load test with concurrent users.

        Args:
            num_users: Number of concurrent threads
            requests_per_user: Number of requests per thread

        Returns:
            LoadTestSummary with statistics
        """
        total_requests = num_users * requests_per_user

        print(f"Starting load test:")
        print(f"  Endpoint: {self.full_url}")
        print(f"  Concurrent users: {num_users}")
        print(f"  Requests per user: {requests_per_user}")
        print(f"  Total requests: {total_requests}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        start_time = time.perf_counter()

        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_users, thread_name_prefix="user_"
        ) as executor:
            # Submit all workers
            futures = [
                executor.submit(self.run_worker, requests_per_user)
                for _ in range(num_users)
            ]

            # Wait for all to complete
            concurrent.futures.wait(futures)

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        # Calculate statistics
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        latencies = [r.latency_ms for r in successful] if successful else []

        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p50_latency = statistics.median(latencies)

            # Calculate percentiles
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p95_latency = (
                sorted_latencies[p95_idx]
                if p95_idx < len(sorted_latencies)
                else max_latency
            )
            p99_latency = (
                sorted_latencies[p99_idx]
                if p99_idx < len(sorted_latencies)
                else max_latency
            )

            # Calculate latency degradation (first half vs second half)
            mid_point = len(latencies) // 2
            if mid_point > 0:
                first_half_avg = statistics.mean(latencies[:mid_point])
                second_half_avg = statistics.mean(latencies[mid_point:])
                latency_degradation = (
                    (second_half_avg - first_half_avg) / first_half_avg
                ) * 100
            else:
                latency_degradation = 0.0
        else:
            avg_latency = 0.0
            min_latency = 0.0
            max_latency = 0.0
            p50_latency = 0.0
            p95_latency = 0.0
            p99_latency = 0.0
            latency_degradation = 0.0

        throughput = total_requests / total_duration if total_duration > 0 else 0

        return LoadTestSummary(
            total_requests=total_requests,
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput,
            total_duration_sec=total_duration,
            latency_degradation_pct=latency_degradation,
        )

    def print_summary(self, summary: LoadTestSummary):
        """Print load test results."""
        print("=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)

        print(f"\n📊 Request Statistics:")
        print(f"  Total requests:     {summary.total_requests}")
        print(
            f"  Successful:         {summary.successful_requests} ({summary.successful_requests / summary.total_requests * 100:.1f}%)"
        )
        print(
            f"  Failed:             {summary.failed_requests} ({summary.failed_requests / summary.total_requests * 100:.1f}%)"
        )

        print(f"\n⏱️  Latency Statistics (ms):")
        print(f"  Average:            {summary.avg_latency_ms:.2f}")
        print(f"  Median (p50):       {summary.p50_latency_ms:.2f}")
        print(f"  Min:                {summary.min_latency_ms:.2f}")
        print(f"  Max:                {summary.max_latency_ms:.2f}")
        print(f"  p95:                {summary.p95_latency_ms:.2f}")
        print(f"  p99:                {summary.p99_latency_ms:.2f}")

        print(f"\n🚀 Performance:")
        print(f"  Throughput:         {summary.throughput_rps:.2f} requests/sec")
        print(f"  Total duration:     {summary.total_duration_sec:.2f} seconds")

        print(f"\n📈 Latency Degradation:")
        if summary.latency_degradation_pct > 5:
            print(
                f"  ⚠️  Degradation: {summary.latency_degradation_pct:.2f}% (second half slower)"
            )
        elif summary.latency_degradation_pct < -5:
            print(
                f"  ✅ Improvement: {abs(summary.latency_degradation_pct):.2f}% (second half faster)"
            )
        else:
            print(
                f"  ✅ Stable: {summary.latency_degradation_pct:.2f}% (no significant change)"
            )

        # Print error details if any
        if summary.failed_requests > 0:
            print(f"\n❌ Error Details:")
            error_counts = {}
            for r in self.results:
                if not r.success:
                    msg = r.error_message or f"HTTP {r.status_code}"
                    error_counts[msg] = error_counts.get(msg, 0) + 1

            for error, count in error_counts.items():
                print(f"  {error}: {count}")

        print("\n" + "=" * 60)

        # Return JSON for programmatic use
        return {
            "avg_latency_ms": summary.avg_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "throughput_rps": summary.throughput_rps,
            "latency_degradation_pct": summary.latency_degradation_pct,
        }


def check_api_running(base_url: str) -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load test СИРЕНА API with concurrent users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test health endpoint with 10 users, 10 requests each
  python3 tests/load_test.py --endpoint /health --users 10 --requests 10
  
  # Test forecast endpoint with higher load
  python3 tests/load_test.py --endpoint /forecast/quick --users 10 --requests 20
  
  # Test custom API server
  python3 tests/load_test.py --url http://localhost:8001 --endpoint /health --users 10
        """,
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/health",
        help="API endpoint to test (default: /health)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        dest="requests_per_user",
        help="Number of requests per user (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check if API is running before starting test",
    )

    args = parser.parse_args()

    # Check API if requested
    if args.check_api:
        print(f"Checking if API is running at {args.url}...")
        if not check_api_running(args.url):
            print(f"❌ ERROR: API is not running at {args.url}")
            print(f"   Try starting the API: uvicorn api.main:app --port 8000")
            sys.exit(1)
        print("✅ API is running\n")

    # Create load tester
    tester = LoadTester(args.url, args.endpoint)

    # Run load test
    try:
        summary = tester.run_load_test(
            num_users=args.users,
            requests_per_user=args.requests_per_user,
        )

        # Print results
        result_json = tester.print_summary(summary)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result_json, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

        # Exit with error code if any failures
        sys.exit(0 if summary.failed_requests == 0 else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR during load test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
