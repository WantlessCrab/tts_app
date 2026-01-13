"""
sum_finder.py
-------------
Standalone helper script.

CONFIGURATION:
    Edit INPUT_VALUES and TARGET_VALUES below.

FEATURES:
    - Accepts any amount of input numbers.
    - Accepts any number of target output values.
    - Handles formats:
        * Currency: $2,450.32
        * Comma thousands: 1,234,567.89
        * Decimals and integers
        * Fractions: 1/2, 3/4, 2 1/8, etc.
        * Mixed formats in the same list
    - Finds:
        1. Exact-match subsets for each target
        2. If no exact match: closest subset and delta
"""

import itertools
import re
from fractions import Fraction

# ---------------------------------------------------------
# CONFIG: EDIT THESE LISTS DIRECTLY
# ---------------------------------------------------------
INPUT_VALUES = [1992.00, 1698.75, 7612.50, 4173.75, 1677.00, 7285.00, 168.75, 2812.50, 900.00, 6886.00]

TARGET_VALUES = [
    "11,005"
]


# ---------------------------------------------------------

def parse_number(raw):
    """Convert arbitrary numeric formats to float."""

    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()

    # Remove currency symbols and whitespace
    s = s.replace("$", "").replace("€", "").replace("£", "")
    s = s.replace(",", "")  # remove commas in thousands

    # Mixed number like "2 1/4"
    if re.match(r"^\d+\s+\d+/\d+$", s):
        whole, frac = s.split()
        return float(Fraction(frac) + int(whole))

    # Pure fraction like "3/4"
    if re.match(r"^\d+/\d+$", s):
        return float(Fraction(s))

    # Integer or float
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid numeric format: {raw}")


def find_best_subset(inputs, target):
    """Search for exact or closest subset."""
    best_subset = None
    best_delta = float("inf")
    exact_matches = []

    for r in range(1, len(inputs) + 1):
        for subset in itertools.combinations(inputs, r):
            s = sum(subset)
            delta = abs(s - target)

            if delta == 0:
                exact_matches.append(subset)

            if delta < best_delta:
                best_delta = delta
                best_subset = subset

    return exact_matches, best_subset, best_delta


def main():
    # Normalize all input values
    numbers = [parse_number(v) for v in INPUT_VALUES]
    targets = [parse_number(v) for v in TARGET_VALUES]

    print("\n===============================")
    print("  SUM FINDER RESULTS")
    print("===============================\n")

    for target in targets:
        print(f"--- Target: {target} ---")

        exact, best, delta = find_best_subset(numbers, target)

        if exact:
            print("Exact matches:")
            for subset in exact:
                print(f"  {subset}   sum={sum(subset)}")
        else:
            print("No exact match found.")
            print(f"Closest subset:\n  {best}")
            print(f"Subset sum: {sum(best)}")
            print(f"Delta:      {delta}")

        print()

    print("Done.\n")


if __name__ == "__main__":
    main()