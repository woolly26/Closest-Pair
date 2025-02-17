import math
import time
import random
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt

# Define a 2D point as a tuple of two floats.
Point = Tuple[float, float]


def euclidean_distance(p1: Point, p2: Point) -> float:

    return math.dist(p1, p2)


def brute_force_closest_pair(points: List[Point]) -> Tuple[float, Tuple[Optional[Point], Optional[Point]]]:

    n = len(points)
    if n < 2:
        return float('inf'), (None, None)

    min_dist = float('inf')
    closest_pair: Tuple[Optional[Point], Optional[Point]] = (None, None)

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            if d < min_dist:
                min_dist = d
                closest_pair = (points[i], points[j])
                if min_dist == 0:  # Early exit if duplicate points found.
                    return 0.0, closest_pair
    return min_dist, closest_pair


def closest_pair_dnc(points: List[Point]) -> Tuple[float, Tuple[Optional[Point], Optional[Point]]]:

    n = len(points)
    if n < 2:
        return float('inf'), (None, None)

    # Sort points by x and y coordinates.
    px = sorted(points, key=lambda p: p[0])
    py = sorted(points, key=lambda p: p[1])
    return _closest_pair_rec(px, py)


def _closest_pair_rec(px: List[Point], py: List[Point]) -> Tuple[float, Tuple[Optional[Point], Optional[Point]]]:

    n = len(px)
    if n <= 3:
        return brute_force_closest_pair(px)

    mid = n // 2
    mid_x = px[mid][0]

    # Divide the points into left and right halves.
    px_left = px[:mid]
    px_right = px[mid:]

    # Partition py into points on the left and right of the midpoint.
    py_left, py_right = [], []
    for p in py:
        if p[0] <= mid_x:
            py_left.append(p)
        else:
            py_right.append(p)

    d_left, pair_left = _closest_pair_rec(px_left, py_left)
    d_right, pair_right = _closest_pair_rec(px_right, py_right)

    if d_left < d_right:
        d_min = d_left
        closest_pair = pair_left
    else:
        d_min = d_right
        closest_pair = pair_right

    # Build a vertical strip around the midpoint with width d_min.
    strip = [p for p in py if abs(p[0] - mid_x) < d_min]
    d_strip, pair_strip = _closest_in_strip(strip, d_min)
    if d_strip < d_min:
        return d_strip, pair_strip
    return d_min, closest_pair


def _closest_in_strip(strip: List[Point], d_min: float) -> Tuple[float, Tuple[Optional[Point], Optional[Point]]]:

    min_dist = d_min
    closest_pair: Tuple[Optional[Point], Optional[Point]] = (None, None)
    n = len(strip)

    for i in range(n):
        j = i + 1
        while j < n and (strip[j][1] - strip[i][1]) < min_dist:
            d = euclidean_distance(strip[i], strip[j])
            if d < min_dist:
                min_dist = d
                closest_pair = (strip[i], strip[j])
                if min_dist == 0:
                    return 0.0, closest_pair
            j += 1
    return min_dist, closest_pair


def benchmark_algorithms():

    test_sizes = [100, 500, 1000, 2000, 5000]
    print("Benchmarking Brute Force vs. Divide & Conquer:")
    print(f"{'n':>6}  {'BF Time (s)':>12}  {'DnC Time (s)':>12}")

    for n in test_sizes:
        points = [(random.uniform(0, 10000), random.uniform(0, 10000)) for _ in range(n)]

        start_time = time.perf_counter()
        bf_dist, bf_pair = brute_force_closest_pair(points)
        bf_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        dnc_dist, dnc_pair = closest_pair_dnc(points)
        dnc_time = time.perf_counter() - start_time

        print(f"{n:>6}  {bf_time:12.6f}  {dnc_time:12.6f}")


def main():

    random.seed(0)  # For reproducibility
    num_points = 20
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(num_points)]

    # Compute closest pair using brute force.
    bf_distance, bf_pair = brute_force_closest_pair(points)
    print(f"[Brute Force] distance = {bf_distance:.4f}, pair = {bf_pair}")

    # Compute closest pair using divide and conquer.
    dnc_distance, dnc_pair = closest_pair_dnc(points)
    print(f"[Divide & Conquer] distance = {dnc_distance:.4f}, pair = {dnc_pair}")

    # Plot the points and highlight the closest pair (from divide & conquer).
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, color='blue', label='Points')

    if dnc_pair[0] is not None and dnc_pair[1] is not None:
        p1, p2 = dnc_pair
        plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], color='red', label='Closest Pair')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linestyle='--')

    plt.title("Closest Pair of Points (Divide & Conquer)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Run benchmark tests.
    benchmark_algorithms()


if __name__ == "__main__":
    main()
