#!/usr/bin/env python3
"""
Main Script to Run Multiple Topic Model Evaluations
====================================================

This script runs multiple topic modeling experiments in sequence:
- 3 BERTopic models (one for each embedding: camembert, e5, mpnet)
- 4 LDA models (different n-gram configurations)
- 1 IRAMUTEQ evaluation

Designed to be run in a tmux session for long-running jobs.

Usage:
    tmux new -s topic_models
    python main_run_multiple_evaluation.py
    # Detach: Ctrl+B, D
    # Reattach: tmux attach -t topic_models
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = PROJECT_DIR / "results"
LOG_DIR = PROJECT_DIR / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

# Experiment configurations
BERTOPIC_EXPERIMENTS = [
    {
        'name': 'BERTopic_camembert',
        'embedding': 'camembert',
        'clusters': 20,
        'compute_embeddings': False,  # Compute for first run
    },
    {
        'name': 'BERTopic_e5',
        'embedding': 'e5',
        'clusters': 20,
        'compute_embeddings': True,
    },
    {
        'name': 'BERTopic_mpnet',
        'embedding': 'mpnet',
        'clusters': 20,
        'compute_embeddings': True,
    },
]

LDA_EXPERIMENTS = [
    {
        'name': 'LDA_bigram_only',
        'ngrams': 'bigram_only',
        'topics': 20,
        'passes': 15,
        'iterations': 400,
    },
    {
        'name': 'LDA_trigram_only',
        'ngrams': 'trigram_only',
        'topics': 20,
        'passes': 15,
        'iterations': 400,
    },
    {
        'name': 'LDA_ngrams_only',
        'ngrams': 'ngrams_only',  # bigrams + trigrams, no unigrams
        'topics': 20,
        'passes': 15,
        'iterations': 400,
    },
    {
        'name': 'LDA_all',
        'ngrams': 'both',  # unigrams + bigrams + trigrams
        'topics': 20,
        'passes': 15,
        'iterations': 400,
    },
]

IRAMUTEQ_EXPERIMENT = {
    'name': 'IRAMUTEQ',
    'min_docs_artist': 10,
}


def log_message(message: str, log_file=None):
    """Print and optionally log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')


def run_command(cmd: list, name: str, log_file: Path) -> tuple:
    """
    Run a command and capture output.
    Returns (success: bool, duration: float)
    """
    log_message(f"Starting: {name}", log_file)
    log_message(f"Command: {' '.join(cmd)}", log_file)

    start_time = time.time()

    try:
        # Run command and capture output
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )

        duration = time.time() - start_time

        # Log output
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"\nSTDERR:\n{result.stderr}\n")
            f.write(f"{'='*60}\n")

        if result.returncode == 0:
            log_message(f"SUCCESS: {name} completed in {duration:.1f}s", log_file)
            return True, duration
        else:
            log_message(f"FAILED: {name} (return code {result.returncode})", log_file)
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        log_message(f"TIMEOUT: {name} exceeded 2 hour limit", log_file)
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        log_message(f"ERROR: {name} - {str(e)}", log_file)
        return False, duration


def run_bertopic_experiment(config: dict, log_file: Path) -> tuple:
    """Run a BERTopic experiment."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "build_and_evaluate_bertopic.py"),
        '--embedding', config['embedding'],
        '--clusters', str(config['clusters']),
    ]

    if config.get('compute_embeddings', False):
        cmd.append('--compute-embeddings')

    # Add common parameters
    cmd.extend([
        '--num-words', '30',
        '--top-artists-topic', '20',
        '--top-artists-heatmap', '50',
    ])

    return run_command(cmd, config['name'], log_file)


def run_lda_experiment(config: dict, log_file: Path) -> tuple:
    """Run an LDA experiment."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "build_and_evaluate_LDA.py"),
        '--topics', str(config['topics']),
        '--passes', str(config['passes']),
        '--iterations', str(config['iterations']),
        '--ngrams', config['ngrams'],
    ]

    # Add common parameters
    cmd.extend([
        '--num-words', '30',
        '--top-artists-topic', '20',
        '--top-artists-heatmap', '50',
    ])

    return run_command(cmd, config['name'], log_file)


def run_iramuteq_experiment(config: dict, log_file: Path) -> tuple:
    """Run IRAMUTEQ evaluation."""
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "evaluate_iramuteq.py"),
        '--min-docs-artist', str(config['min_docs_artist']),
        '--top-artists-topic', '20',
        '--top-artists-heatmap', '50',
    ]

    return run_command(cmd, config['name'], log_file)


def print_summary(results: list, total_duration: float, log_file: Path):
    """Print and log a summary of all experiments."""
    summary = [
        "",
        "=" * 70,
        "EXPERIMENT SUMMARY",
        "=" * 70,
        "",
    ]

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    summary.append(f"Total experiments: {len(results)}")
    summary.append(f"Successful: {successful}")
    summary.append(f"Failed: {failed}")
    summary.append(f"Total duration: {total_duration/60:.1f} minutes")
    summary.append("")
    summary.append("-" * 70)
    summary.append(f"{'Experiment':<30} {'Status':<10} {'Duration':<15}")
    summary.append("-" * 70)

    for r in results:
        status = "SUCCESS" if r['success'] else "FAILED"
        duration_str = f"{r['duration']:.1f}s"
        summary.append(f"{r['name']:<30} {status:<10} {duration_str:<15}")

    summary.append("-" * 70)
    summary.append("")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(log_file, 'a') as f:
        f.write(summary_text)


def main():
    """Main function to run all experiments."""
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"batch_run_{timestamp}.log"

    log_message("=" * 70, log_file)
    log_message("STARTING BATCH TOPIC MODELING EVALUATION", log_file)
    log_message("=" * 70, log_file)
    log_message(f"Log file: {log_file}", log_file)
    log_message(f"Project directory: {PROJECT_DIR}", log_file)
    log_message("", log_file)

    # List all experiments
    log_message("Planned experiments:", log_file)
    log_message("  BERTopic:", log_file)
    for exp in BERTOPIC_EXPERIMENTS:
        log_message(f"    - {exp['name']}", log_file)
    log_message("  LDA:", log_file)
    for exp in LDA_EXPERIMENTS:
        log_message(f"    - {exp['name']}", log_file)
    log_message("  IRAMUTEQ:", log_file)
    log_message(f"    - {IRAMUTEQ_EXPERIMENT['name']}", log_file)
    log_message("", log_file)

    total_experiments = len(BERTOPIC_EXPERIMENTS) + len(LDA_EXPERIMENTS) + 1
    log_message(f"Total experiments to run: {total_experiments}", log_file)
    log_message("", log_file)

    results = []
    total_start = time.time()

    # Run IRAMUTEQ first (fastest, no model building)
    log_message("=" * 70, log_file)
    log_message("PHASE 1: IRAMUTEQ EVALUATION", log_file)
    log_message("=" * 70, log_file)

    success, duration = run_iramuteq_experiment(IRAMUTEQ_EXPERIMENT, log_file)
    results.append({
        'name': IRAMUTEQ_EXPERIMENT['name'],
        'success': success,
        'duration': duration,
    })
    log_message("", log_file)

    # Run LDA experiments
    log_message("=" * 70, log_file)
    log_message("PHASE 2: LDA EXPERIMENTS", log_file)
    log_message("=" * 70, log_file)

    for i, config in enumerate(LDA_EXPERIMENTS, 1):
        log_message(f"\n--- LDA Experiment {i}/{len(LDA_EXPERIMENTS)} ---", log_file)
        success, duration = run_lda_experiment(config, log_file)
        results.append({
            'name': config['name'],
            'success': success,
            'duration': duration,
        })

    log_message("", log_file)

    # Run BERTopic experiments
    log_message("=" * 70, log_file)
    log_message("PHASE 3: BERTOPIC EXPERIMENTS", log_file)
    log_message("=" * 70, log_file)

    for i, config in enumerate(BERTOPIC_EXPERIMENTS, 1):
        log_message(f"\n--- BERTopic Experiment {i}/{len(BERTOPIC_EXPERIMENTS)} ---", log_file)
        success, duration = run_bertopic_experiment(config, log_file)
        results.append({
            'name': config['name'],
            'success': success,
            'duration': duration,
        })

    total_duration = time.time() - total_start

    # Print summary
    print_summary(results, total_duration, log_file)

    log_message(f"Full log saved to: {log_file}", log_file)
    log_message("", log_file)
    log_message("BATCH RUN COMPLETE", log_file)

    # Return exit code based on success
    if all(r['success'] for r in results):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
