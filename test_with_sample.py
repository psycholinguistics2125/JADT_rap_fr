#!/usr/bin/env python3
"""
Test Script for Topic Modeling Evaluation
==========================================
Runs LDA, BERTopic, and IRAMUTEQ evaluations with a sample size
to verify that the refactored shared functions work correctly.

Usage:
    python test_with_sample.py --sample 500
    python test_with_sample.py --sample 1000 --skip-bertopic
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()


def run_command(cmd: list, name: str) -> tuple:
    """
    Run a command and return (success, duration, error_message).
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"SUCCESS: {name} completed in {duration:.1f}s")
            # Print last 20 lines of output
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 20:
                print("\n... (truncated output) ...")
            for line in output_lines[-20:]:
                print(f"  {line}")
            return True, duration, None
        else:
            print(f"FAILED: {name} (return code {result.returncode})")
            print("\nSTDERR:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            print("\nSTDOUT (last 50 lines):")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-50:]:
                print(f"  {line}")
            return False, duration, result.stderr

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"TIMEOUT: {name} exceeded 10 minute limit")
        return False, duration, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR: {name} - {str(e)}")
        return False, duration, str(e)


def test_iramuteq(sample_size: int) -> tuple:
    """Test IRAMUTEQ evaluation."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "evaluate_iramuteq.py"),
        '--sample', str(sample_size),
        '--min-docs-artist', '5',  # Lower threshold for small sample
    ]
    return run_command(cmd, "IRAMUTEQ Evaluation")


def test_lda(sample_size: int) -> tuple:
    """Test LDA evaluation."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "build_and_evaluate_LDA.py"),
        '--sample', str(sample_size),
        '--topics', '10',  # Fewer topics for faster testing
        '--passes', '5',   # Fewer passes for faster testing
        '--iterations', '100',
        '--ngrams', 'bigrams',
        '--no-pyldavis',  # Skip pyLDAvis for speed
    ]
    return run_command(cmd, "LDA Evaluation")


def test_bertopic(sample_size: int, skip_keybert: bool = False,
                  skip_html: bool = False) -> tuple:
    """Test BERTopic evaluation."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "build_and_evaluate_bertopic.py"),
        '--sample', str(sample_size),
        '--clusters', '10',  # Fewer clusters for faster testing
        '--no-openai',  # Skip OpenAI for speed
        # Note: embeddings auto-computed for sample runs
    ]
    if skip_keybert:
        cmd.append('--no-keybert')
    if skip_html:
        cmd.append('--no-interactive-html')
    return run_command(cmd, "BERTopic Evaluation")


def main():
    parser = argparse.ArgumentParser(
        description='Test topic modeling scripts with a sample'
    )
    parser.add_argument(
        '--sample', type=int, default=500,
        help='Sample size for testing (default: 500)'
    )
    parser.add_argument(
        '--skip-lda', action='store_true',
        help='Skip LDA test'
    )
    parser.add_argument(
        '--skip-bertopic', action='store_true',
        help='Skip BERTopic test'
    )
    parser.add_argument(
        '--skip-iramuteq', action='store_true',
        help='Skip IRAMUTEQ test'
    )
    parser.add_argument(
        '--skip-keybert', action='store_true',
        help='Skip KeyBERTInspired representation (enabled by default)'
    )
    parser.add_argument(
        '--skip-html', action='store_true',
        help='Skip interactive HTML visualization (enabled by default)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("TOPIC MODELING TEST SUITE")
    print("="*60)
    print(f"Sample size: {args.sample}")
    print(f"Project directory: {PROJECT_DIR}")

    results = []
    total_start = time.time()

    # Test IRAMUTEQ (fastest, no model building)
    if not args.skip_iramuteq:
        success, duration, error = test_iramuteq(args.sample)
        results.append(('IRAMUTEQ', success, duration, error))

    # Test LDA
    if not args.skip_lda:
        success, duration, error = test_lda(args.sample)
        results.append(('LDA', success, duration, error))

    # Test BERTopic
    if not args.skip_bertopic:
        success, duration, error = test_bertopic(
            args.sample,
            skip_keybert=args.skip_keybert,
            skip_html=args.skip_html
        )
        results.append(('BERTopic', success, duration, error))

    total_duration = time.time() - total_start

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, success, duration, error in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name:15s}: {status:4s} ({duration:.1f}s)")
        if not success:
            all_passed = False

    print(f"\nTotal time: {total_duration:.1f}s")

    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
