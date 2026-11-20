#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path


def paired_summary(observations, first_mode="vector", second_mode="hybrid", seed=20261118, resamples=10000):
    groups = {first_mode: {}, second_mode: {}}
    if first_mode == second_mode or resamples < 1:
        raise ValueError("Compare two different modes with at least one resample.")
    for observation in observations:
        mode = observation.get("mode")
        if mode not in groups:
            continue
        identifier = observation.get("query_id", observation.get("claim_id"))
        if not isinstance(identifier, str) or not identifier or identifier in groups[mode]:
            raise ValueError("Comparison contains a missing or duplicate query ID.")
        rank = observation.get("first_relevant_rank", observation.get("rank"))
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError("Comparison contains an invalid relevance rank.")
        relevant = observation.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError("Comparison requires relevance labels for each observation.")
        groups[mode][identifier] = {"hit": rank == 1, "relevant": relevant}
    if not groups[first_mode] or groups[first_mode].keys() != groups[second_mode].keys():
        raise ValueError("The two modes must contain the same nonempty set of query IDs.")
    differences = []
    counts = {"both_hit": 0, "first_only": 0, "second_only": 0, "neither_hit": 0}
    for identifier in sorted(groups[first_mode]):
        first, second = groups[first_mode][identifier], groups[second_mode][identifier]
        if set(first["relevant"]) != set(second["relevant"]):
            raise ValueError("Relevance labels differ between modes for the same query.")
        key = ("both_hit" if first["hit"] and second["hit"] else
               "first_only" if first["hit"] else "second_only" if second["hit"] else "neither_hit")
        counts[key] += 1
        differences.append(int(second["hit"]) - int(first["hit"]))
    rng = random.Random(seed)
    size = len(differences)
    bootstrapped = sorted(sum(rng.choice(differences) for unused in range(size)) / size * 100
                         for unused in range(resamples))
    lower = bootstrapped[max(0, math.ceil(resamples * 0.025) - 1)]
    upper = bootstrapped[min(resamples - 1, math.ceil(resamples * 0.975) - 1)]
    return {
        "queries": size, "first_mode": first_mode, "second_mode": second_mode,
        "paired_hit_at_1_counts": counts,
        "second_minus_first_percentage_points": round(sum(differences) / size * 100, 2),
        "paired_bootstrap_95_percentile_interval_percentage_points": [round(lower, 2), round(upper, 2)],
        "resamples": resamples, "seed": seed,
    }


def run(arguments):
    comparisons = []
    for path in arguments.reports:
        data = Path(path).read_bytes()
        report = json.loads(data)
        result = paired_summary(report["observations"], arguments.first, arguments.second,
                                arguments.seed, arguments.resamples)
        result["report_file"] = Path(path).name
        result["report_sha256"] = hashlib.sha256(data).hexdigest()
        comparisons.append(result)
    output = {
        "method": "Paired hit@1 differences; resample the same query pairs with replacement, using a fixed seed and percentile endpoints.",
        "limitations": "These nonrandom, small datasets and constructed candidate pools do not justify generalizing the interval to a production population. Resampling captures query-to-query variation only, not labeling, model, pool-selection, or run-to-run uncertainty.",
        "comparisons": comparisons,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Calculate paired hit@1 differences and a descriptive paired bootstrap interval.")
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--first", default="vector")
    parser.add_argument("--second", default="hybrid")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
        return 0
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(f"Paired comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
