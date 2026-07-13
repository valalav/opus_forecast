"""
Load testing script for Opus Edge Lab API.

Task 564: Simple load test with 10 concurrent users.
Measures latency degradation under concurrent load.
"""

import time
import threading
import statistics
import requests
from typing import List, Dict, Any
from datetime import datetime
import argparse
import json


BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


class LoadTestResult:
    """Store results from a single request."""

    def __init__(
        self,
        user_id: int,
        request_id: int,
        success: bool,
        latency_ms: float,
        error: str = None,
    ):
        self.user_id = user_id
        self.request_id = request_id
        self.success = success
        self.latency_ms = latency_ms
        self.error = error
        self.timestamp = datetime.now()


class LoadTestWorker:
    """Simulates a single user making requests."""

    def __init__(
        self,
        user_id: int,
        num_requests: int,
        results: List[LoadTestResult],
        delay_ms: float = 0,
    ):
        self.user_id = user_id
        self.num_requests = num_requests
        self.results = results
        self.delay_ms = delay_ms

    def make_request(self) -> LoadTestResult:
        """Make a single request to health endpoint."""
        request_id = len([r for r in self.results if r.user_id == self.user_id])

        try:
            start_time = time.perf_counter()
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            success = response.status_code == 200

            if success:
                return LoadTestResult(
                    user_id=self.user_id,
                    request_id=request_id,
                    success=True,
                    latency_ms=latency_ms,
                )
            else:
                return LoadTestResult(
                    user_id=self.user_id,
                    request_id=request_id,
                    success=False,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}",
                )

        except requests.exceptions.Timeout:
            return LoadTestResult(
                user_id=self.user_id,
                request_id=request_id,
                success=False,
                latency_ms=5000,
                error="Timeout",
            )
        except Exception as e:
            return LoadTestResult(
                user_id=self.user_id,
                request_id=request_id,
                success=False,
                latency_ms=0,
                error=str(e),
            )

    def run(self):
        """Run all requests for this user."""
        for i in range(self.num_requests):
            result = self.make_request()
            self.results.append(result)

            # Add small delay between requests (realistic user behavior)
            if self.delay_ms > 0:
                time.sleep(self.delay_ms / 1000)


def calculate_metrics(results: List[LoadTestResult]) -> Dict[str, Any]:
    """Calculate load test metrics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        return {
            "total_requests": len(results),
            "successful": 0,
            "failed": len(failed),
            "success_rate": 0.0,
            "avg_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
            "errors": [r.error for r in failed],
        }

    latencies = [r.latency_ms for r in successful]

    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100,
        "avg_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_latency_ms": sorted_latencies[int(n * 0.5)],
        "p95_latency_ms": sorted_latencies[int(n * 0.95)],
        "p99_latency_ms": sorted_latencies[int(n * 0.99)],
        "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "errors": [r.error for r in failed],
    }


def detect_degradation(
    results: List[LoadTestResult], window_size: int = 10
) -> Dict[str, Any]:
    """Analyze latency degradation over time."""
    successful = [r for r in results if r.success]

    if len(successful) < window_size * 2:
        return {
            "degradation_detected": False,
            "reason": "Insufficient data",
            "interpretation": "Too few requests to analyze degradation",
            "first_half_avg_ms": 0,
            "second_half_avg_ms": 0,
            "degradation_pct": 0,
        }

    # Split into first half and second half
    mid = len(successful) // 2
    first_half = successful[:mid]
    second_half = successful[mid:]

    first_avg = statistics.mean([r.latency_ms for r in first_half])
    second_avg = statistics.mean([r.latency_ms for r in second_half])

    degradation_pct = (
        ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
    )

    return {
        "degradation_detected": degradation_pct > 10,  # 10% threshold
        "first_half_avg_ms": first_avg,
        "second_half_avg_ms": second_avg,
        "degradation_pct": degradation_pct,
        "interpretation": (
            f"Latency increased by {degradation_pct:.1f}% "
            f"({first_avg:.2f}ms -> {second_avg:.2f}ms)"
        ),
    }


def run_load_test(
    num_users: int = 10, requests_per_user: int = 20, delay_ms: float = 100
) -> Dict[str, Any]:
    """
    Run load test with concurrent users.

    Args:
        num_users: Number of concurrent users
        requests_per_user: Number of requests per user
        delay_ms: Delay between requests in milliseconds

    Returns:
        Dictionary with test results and metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Load Test Configuration")
    print(f"{'=' * 60}")
    print(f"Target URL: {HEALTH_ENDPOINT}")
    print(f"Concurrent Users: {num_users}")
    print(f"Requests per User: {requests_per_user}")
    print(f"Total Requests: {num_users * requests_per_user}")
    print(f"Delay between requests: {delay_ms}ms")
    print(f"{'=' * 60}\n")

    # Check if server is running
    try:
        requests.get(HEALTH_ENDPOINT, timeout=2)
        print("✓ Server is running and accessible")
    except requests.exceptions.ConnectionError:
        print("✗ ERROR: Server is not running. Start with: uvicorn api.main:app")
        return {"error": "Server not running"}
    except Exception as e:
        print(f"✗ ERROR: Cannot connect to server: {e}")
        return {"error": str(e)}

    # Shared results list (thread-safe for append operations)
    results: List[LoadTestResult] = []
    results_lock = threading.Lock()

    # Create worker threads
    threads = []
    print(f"Starting {num_users} concurrent users...\n")

    start_time = time.perf_counter()

    for user_id in range(num_users):
        worker = LoadTestWorker(
            user_id=user_id,
            num_requests=requests_per_user,
            results=results,
            delay_ms=delay_ms,
        )
        thread = threading.Thread(target=worker.run, name=f"User-{user_id}")
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    print(f"All requests completed in {total_duration:.2f}s\n")

    # Calculate metrics
    metrics = calculate_metrics(results)
    degradation = detect_degradation(results)

    # Add test metadata
    test_metadata = {
        "test_config": {
            "num_users": num_users,
            "requests_per_user": requests_per_user,
            "total_requests": num_users * requests_per_user,
            "delay_ms": delay_ms,
        },
        "duration_seconds": total_duration,
        "requests_per_second": (num_users * requests_per_user) / total_duration,
    }

    return {"metadata": test_metadata, "metrics": metrics, "degradation": degradation}


def print_report(results: Dict[str, Any]):
    """Print formatted load test report."""
    if "error" in results:
        return

    print(f"\n{'=' * 60}")
    print(f"LOAD TEST RESULTS")
    print(f"{'=' * 60}")

    # Test Configuration
    config = results["metadata"]["test_config"]
    print(f"\nTest Configuration:")
    print(f"  Duration: {results['metadata']['duration_seconds']:.2f}s")
    print(f"  Throughput: {results['metadata']['requests_per_second']:.2f} req/s")
    print(f"  Users: {config['num_users']}")
    print(f"  Total Requests: {config['total_requests']}")

    # Success Metrics
    metrics = results["metrics"]
    print(f"\nSuccess Metrics:")
    print(f"  Total Requests: {metrics['total_requests']}")
    print(f"  Successful: {metrics['successful']}")
    print(f"  Failed: {metrics['failed']}")
    print(f"  Success Rate: {metrics['success_rate']:.1f}%")

    # Latency Metrics
    print(f"\nLatency Metrics (successful requests):")
    print(f"  Average: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  Min: {metrics['min_latency_ms']:.2f}ms")
    print(f"  Max: {metrics['max_latency_ms']:.2f}ms")
    print(f"  Std Dev: {metrics['std_latency_ms']:.2f}ms")
    print(f"  P50 (Median): {metrics['p50_latency_ms']:.2f}ms")
    print(f"  P95: {metrics['p95_latency_ms']:.2f}ms")
    print(f"  P99: {metrics['p99_latency_ms']:.2f}ms")

    # Degradation Analysis
    degradation = results["degradation"]
    print(f"\nLatency Degradation:")
    if degradation["degradation_detected"]:
        print(f"  ⚠️  DEGRADATION DETECTED: {degradation['interpretation']}")
    else:
        print(f"  ✓ No significant degradation: {degradation['interpretation']}")

    # Errors
    if metrics["errors"]:
        print(f"\nErrors:")
        for error in set(metrics["errors"]):
            count = metrics["errors"].count(error)
            print(f"  - {error} ({count}x)")

    print(f"\n{'=' * 60}\n")


def save_results(results: Dict[str, Any], output_file: str = None):
    """Save results to JSON file."""
    if output_file is None:
        output_file = (
            f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load test for Opus Edge Lab API")
    parser.add_argument(
        "--users",
        "-u",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--requests",
        "-r",
        type=int,
        default=20,
        help="Number of requests per user (default: 20)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=100,
        help="Delay between requests in ms (default: 100)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default="health",
        help="Endpoint to test: health, models, metrics (default: health)",
    )

    args = parser.parse_args()

    # Update endpoint based on argument
    global HEALTH_ENDPOINT
    if args.endpoint == "health":
        HEALTH_ENDPOINT = f"{BASE_URL}/health"
    elif args.endpoint == "models":
        HEALTH_ENDPOINT = f"{BASE_URL}/models"
    elif args.endpoint == "metrics":
        HEALTH_ENDPOINT = f"{BASE_URL}/metrics"
    else:
        print(f"Unknown endpoint: {args.endpoint}")
        return 1

    # Run load test
    results = run_load_test(
        num_users=args.users, requests_per_user=args.requests, delay_ms=args.delay
    )

    # Print report
    print_report(results)

    # Save results
    if "error" not in results and args.output:
        save_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
