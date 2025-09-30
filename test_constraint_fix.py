#!/usr/bin/env python3
"""
Test the corrected constraint functions vs. the original ones
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Copy the necessary functions from your notebook
def great_circle_distance(lat1, lon1, lat2, lon2):
    """calculates great circle (angular) distance between two geographical coordinates"""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

    x1 = np.cos(lat1_rad) * np.cos(lon1_rad)
    y1 = np.cos(lat1_rad) * np.sin(lon1_rad)
    z1 = np.sin(lat1_rad)

    x2 = np.cos(lat2_rad) * np.cos(lon2_rad)
    y2 = np.cos(lat2_rad) * np.sin(lon2_rad)
    z2 = np.sin(lat2_rad)

    dot_product = x1*x2 + y1*y2 + z1*z2
    angular_distance = np.arccos(max(-1, min(1, dot_product)))
    return angular_distance

def spherical_coverage_overlap(lat1, lon1, lat2, lon2):
    """calculate the overlap to optimise coverage"""
    angular_distance = great_circle_distance(lat1, lon1, lat2, lon2)
    overlap = np.cos(angular_distance/2)
    return max(0, overlap)

# Original constraint functions (problematic)
def constraint_matrix_original(satellite_positions, constraint_type):
    """Original constraint matrix construction (with bugs)"""
    number_satellites = len(satellite_positions)

    constraint_functions = {
        "redundancy": lambda overlap: overlap**2,     # WRONG: rewards high overlap
        "efficiency": lambda overlap: 1 - overlap    # WRONG: creates inverted eigenvalues
    }

    C = np.zeros((number_satellites, number_satellites))

    for i in range(number_satellites):
        for j in range(number_satellites):
            if i == j:
                C[i,j] = 1
            else:
                overlap = spherical_coverage_overlap(satellite_positions[i][0], satellite_positions[i][1],
                                                    satellite_positions[j][0], satellite_positions[j][1])
                C[i,j] = constraint_functions[constraint_type](overlap)

    return C

# Corrected constraint functions (penalty-based)
def constraint_matrix_fixed(satellite_positions, constraint_type):
    """Fixed constraint matrix construction (penalty-based)"""
    number_satellites = len(satellite_positions)

    # CORRECTED: Both represent penalties
    constraint_functions = {
        "efficiency": lambda overlap: overlap,        # Penalty for high overlap (bad for efficiency)
        "redundancy": lambda overlap: 1 - overlap    # Penalty for low overlap (bad for redundancy)
    }

    C = np.zeros((number_satellites, number_satellites))

    for i in range(number_satellites):
        for j in range(number_satellites):
            if i == j:
                C[i,j] = 1
            else:
                overlap = spherical_coverage_overlap(satellite_positions[i][0], satellite_positions[i][1],
                                                    satellite_positions[j][0], satellite_positions[j][1])
                C[i,j] = constraint_functions[constraint_type](overlap)

    return C

def communication_matrix(satellite_positions, optimal_distance=5000, sigma=2500):
    """Communication matrix (unchanged)"""
    number_satellites = len(satellite_positions)
    C = np.zeros((number_satellites, number_satellites))

    for i in range(number_satellites):
        for j in range(number_satellites):
            if i == j:
                C[i,j] = 1
            else:
                angular_distance = great_circle_distance(satellite_positions[i][0], satellite_positions[i][1],
                                                    satellite_positions[j][0], satellite_positions[j][1])
                distance_km = 6371 * angular_distance
                communication_quality = np.exp(-((distance_km - optimal_distance)**2) / (2 * sigma**2))
                C[i,j] = communication_quality

    return C

def combined_matrix(satellite_positions, weights, version="original"):
    """Combine constraint matrices"""
    if version == "original":
        efficiency_matrix = constraint_matrix_original(satellite_positions, "efficiency")
        redundancy_matrix = constraint_matrix_original(satellite_positions, "redundancy")
    else:  # fixed
        efficiency_matrix = constraint_matrix_fixed(satellite_positions, "efficiency")
        redundancy_matrix = constraint_matrix_fixed(satellite_positions, "redundancy")

    signal_matrix = communication_matrix(satellite_positions)

    C = (weights["efficiency"] * efficiency_matrix +
         weights["redundancy"] * redundancy_matrix +
         weights["communication"] * signal_matrix)

    return C

def objective(satellite_positions_flat, weights, version="original"):
    """Calculate eigenvalue objective"""
    np_positions = np.reshape(satellite_positions_flat, (-1, 2))
    C = combined_matrix(np_positions, weights, version)
    eigenvalues = np.linalg.eigvals(C)
    return -np.min(eigenvalues)  # Minimize negative min eigenvalue = maximize min eigenvalue

def optimize_satellites(initial_positions, weights, version="original"):
    """Optimize satellite positions"""
    flat_positions = initial_positions.flatten()
    bounds = [(-90, 90), (-180, 180)] * len(initial_positions)

    result = minimize(objective, flat_positions, args=(weights, version),
                     method='L-BFGS-B', bounds=bounds)

    optimized_positions = np.reshape(result.x, (-1, 2))
    return optimized_positions, result.fun

def test_constraint_fix():
    """Test original vs. fixed constraint functions"""
    print("🧪 TESTING CONSTRAINT FUNCTION FIX")
    print("="*60)

    # Create test constellation - satellites start close together
    initial_positions = np.array([
        [0.0, 0.0],      # Satellite 1 at equator/prime meridian
        [5.0, 5.0]       # Satellite 2 nearby (should spread out for efficiency)
    ])

    # Test with efficiency-heavy weights
    weights = {"efficiency": 0.8, "redundancy": 0.1, "communication": 0.1}

    print(f"Initial satellite positions:")
    print(f"  Sat 1: ({initial_positions[0][0]:.1f}°, {initial_positions[0][1]:.1f}°)")
    print(f"  Sat 2: ({initial_positions[1][0]:.1f}°, {initial_positions[1][1]:.1f}°)")

    # Calculate initial separation
    initial_distance = great_circle_distance(*initial_positions[0], *initial_positions[1])
    initial_separation = initial_distance * 180 / np.pi
    print(f"  Initial separation: {initial_separation:.1f}°")

    print(f"\nWeights: {weights}")
    print(f"Expected behavior: Efficiency optimization should SPREAD satellites apart")

    # Test original constraint functions
    print(f"\n1️⃣ ORIGINAL CONSTRAINT FUNCTIONS:")
    start_time = time.time()
    optimized_original, obj_original = optimize_satellites(initial_positions, weights, "original")
    time_original = time.time() - start_time

    final_distance_orig = great_circle_distance(*optimized_original[0], *optimized_original[1])
    final_separation_orig = final_distance_orig * 180 / np.pi

    print(f"  Final positions:")
    print(f"    Sat 1: ({optimized_original[0][0]:.1f}°, {optimized_original[0][1]:.1f}°)")
    print(f"    Sat 2: ({optimized_original[1][0]:.1f}°, {optimized_original[1][1]:.1f}°)")
    print(f"  Final separation: {final_separation_orig:.1f}°")
    print(f"  Objective value: {-obj_original:.4f}")
    print(f"  Time: {time_original:.3f} seconds")

    # Test fixed constraint functions
    print(f"\n2️⃣ FIXED CONSTRAINT FUNCTIONS:")
    start_time = time.time()
    optimized_fixed, obj_fixed = optimize_satellites(initial_positions, weights, "fixed")
    time_fixed = time.time() - start_time

    final_distance_fixed = great_circle_distance(*optimized_fixed[0], *optimized_fixed[1])
    final_separation_fixed = final_distance_fixed * 180 / np.pi

    print(f"  Final positions:")
    print(f"    Sat 1: ({optimized_fixed[0][0]:.1f}°, {optimized_fixed[0][1]:.1f}°)")
    print(f"    Sat 2: ({optimized_fixed[1][0]:.1f}°, {optimized_fixed[1][1]:.1f}°)")
    print(f"  Final separation: {final_separation_fixed:.1f}°")
    print(f"  Objective value: {-obj_fixed:.4f}")
    print(f"  Time: {time_fixed:.3f} seconds")

    # Analysis
    print(f"\n📊 RESULTS ANALYSIS:")
    print(f"  Initial separation: {initial_separation:.1f}°")
    print(f"  Original method final separation: {final_separation_orig:.1f}°")
    print(f"  Fixed method final separation: {final_separation_fixed:.1f}°")

    separation_change_orig = final_separation_orig - initial_separation
    separation_change_fixed = final_separation_fixed - initial_separation

    print(f"\n  Separation changes:")
    print(f"  Original: {separation_change_orig:+.1f}° ({'MOVED CLOSER' if separation_change_orig < 0 else 'MOVED APART'})")
    print(f"  Fixed: {separation_change_fixed:+.1f}° ({'MOVED CLOSER' if separation_change_fixed < 0 else 'MOVED APART'})")

    print(f"\n✅ SUCCESS CRITERIA:")
    print(f"  Fixed method should move satellites APART for efficiency optimization")
    if separation_change_fixed > separation_change_orig:
        print(f"  ✅ FIXED! Satellites moved farther apart with corrected constraints")
    else:
        print(f"  ❌ Still has issues - need further investigation")

    return {
        'initial_separation': initial_separation,
        'original_final_separation': final_separation_orig,
        'fixed_final_separation': final_separation_fixed,
        'original_objective': -obj_original,
        'fixed_objective': -obj_fixed
    }

def test_redundancy_behavior():
    """Test redundancy optimization behavior"""
    print(f"\n🔄 TESTING REDUNDANCY OPTIMIZATION")
    print("="*50)

    # Start with satellites far apart
    initial_positions = np.array([
        [0.0, 0.0],      # Satellite 1
        [0.0, 90.0]      # Satellite 2 - 90° away
    ])

    # Test with redundancy-heavy weights
    weights = {"efficiency": 0.1, "redundancy": 0.8, "communication": 0.1}

    initial_distance = great_circle_distance(*initial_positions[0], *initial_positions[1])
    initial_separation = initial_distance * 180 / np.pi
    print(f"Initial separation: {initial_separation:.1f}° (satellites far apart)")
    print(f"Expected: Redundancy optimization should PULL satellites together")

    # Test fixed constraint functions for redundancy
    optimized, obj = optimize_satellites(initial_positions, weights, "fixed")

    final_distance = great_circle_distance(*optimized[0], *optimized[1])
    final_separation = final_distance * 180 / np.pi
    separation_change = final_separation - initial_separation

    print(f"Final separation: {final_separation:.1f}°")
    print(f"Change: {separation_change:+.1f}° ({'MOVED CLOSER' if separation_change < 0 else 'MOVED APART'})")

    if separation_change < -10:  # Moved significantly closer
        print(f"✅ Redundancy optimization working correctly!")
    else:
        print(f"⚠️  Redundancy optimization may need adjustment")

if __name__ == "__main__":
    results = test_constraint_fix()
    test_redundancy_behavior()