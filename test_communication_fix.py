#!/usr/bin/env python3
"""
Test the penalty-based communication matrix fix
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

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

# Fixed constraint matrices (penalty-based)
def constraint_matrix_fixed(satellite_positions, constraint_type):
    """Fixed constraint matrix construction (penalty-based)"""
    number_satellites = len(satellite_positions)

    # All represent penalties for poor constraint satisfaction
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

# Original communication matrix (problematic)
def communication_matrix_original(satellite_positions, optimal_distance=5000, sigma=2500):
    """Original communication matrix (rewards good communication)"""
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
                C[i,j] = communication_quality  # REWARD good communication

    return C

# Fixed communication matrix (penalty-based)
def communication_matrix_fixed(satellite_positions, optimal_distance=5000, sigma=2500):
    """Fixed communication matrix (penalizes poor communication)"""
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

                # PENALTY for poor communication
                communication_penalty = 1 - communication_quality
                C[i,j] = communication_penalty

    return C

def combined_matrix(satellite_positions, weights, comm_version="original"):
    """Combine constraint matrices"""
    efficiency_matrix = constraint_matrix_fixed(satellite_positions, "efficiency")
    redundancy_matrix = constraint_matrix_fixed(satellite_positions, "redundancy")

    if comm_version == "original":
        signal_matrix = communication_matrix_original(satellite_positions)
    else:  # fixed
        signal_matrix = communication_matrix_fixed(satellite_positions)

    C = (weights["efficiency"] * efficiency_matrix +
         weights["redundancy"] * redundancy_matrix +
         weights["communication"] * signal_matrix)

    return C

def objective(satellite_positions_flat, weights, comm_version="original"):
    """Calculate eigenvalue objective"""
    np_positions = np.reshape(satellite_positions_flat, (-1, 2))
    C = combined_matrix(np_positions, weights, comm_version)
    eigenvalues = np.linalg.eigvals(C)
    return -np.min(eigenvalues)  # Minimize negative min eigenvalue = maximize min eigenvalue

def optimize_satellites(initial_positions, weights, comm_version="original"):
    """Optimize satellite positions"""
    flat_positions = initial_positions.flatten()
    bounds = [(-89, 89), (-179, 179)] * len(initial_positions)  # Slightly inside boundaries

    result = minimize(objective, flat_positions, args=(weights, comm_version),
                     method='L-BFGS-B', bounds=bounds)

    optimized_positions = np.reshape(result.x, (-1, 2))
    return optimized_positions, result.fun

def test_communication_matrices():
    """Test original vs fixed communication matrices"""
    print("🔍 TESTING COMMUNICATION MATRIX BEHAVIOR")
    print("="*60)

    # Test different satellite configurations
    configs = [
        ('Identical positions', np.array([[0.0, 0.0], [0.0, 0.0]])),
        ('Optimal comm (45°)', np.array([[0.0, 0.0], [0.0, 45.0]])),  # ~5000km apart
        ('Close together', np.array([[0.0, 0.0], [5.0, 5.0]])),
        ('Far apart', np.array([[0.0, 0.0], [0.0, 90.0]])),
    ]

    print("Configuration         | Original Matrix | Fixed Matrix    | Distance")
    print("                     | Min Eigenvalue  | Min Eigenvalue  | (km)")
    print("-" * 70)

    for name, positions in configs:
        # Calculate actual distance
        if not np.allclose(positions[0], positions[1]):
            dist_km = 6371 * great_circle_distance(positions[0][0], positions[0][1],
                                                 positions[1][0], positions[1][1])
        else:
            dist_km = 0

        # Original communication matrix
        comm_orig = communication_matrix_original(positions)
        eig_orig = np.min(np.linalg.eigvals(comm_orig))

        # Fixed communication matrix
        comm_fixed = communication_matrix_fixed(positions)
        eig_fixed = np.min(np.linalg.eigvals(comm_fixed))

        print(f"{name:<20} | {eig_orig:13.3f} | {eig_fixed:13.3f} | {dist_km:6.0f}")

    print()
    print("ANALYSIS:")
    print("- Original: Optimal communication (45°) gives min eigenvalue = 0.000 (WORST)")
    print("- Fixed: Optimal communication (45°) should give better eigenvalue")
    print("- Fixed version should prefer the optimal 5000km separation")

def test_optimization_with_communication_fix():
    """Test optimization with the communication matrix fix"""
    print("\n🚀 TESTING OPTIMIZATION WITH COMMUNICATION FIX")
    print("="*60)

    # Start with satellites close together
    initial_positions = np.array([
        [0.0, 0.0],      # Satellite 1
        [5.0, 5.0]       # Satellite 2 - close together
    ])

    # Test with communication-heavy weights
    weights = {"efficiency": 0.2, "redundancy": 0.2, "communication": 0.6}

    initial_distance = great_circle_distance(*initial_positions[0], *initial_positions[1])
    initial_separation = initial_distance * 180 / np.pi
    print(f"Initial separation: {initial_separation:.1f}°")
    print(f"Optimal communication separation: ~45° (5000km)")
    print(f"Weights: {weights}")

    # Test original communication matrix
    print(f"\n1️⃣ ORIGINAL COMMUNICATION MATRIX:")
    start_time = time.time()
    optimized_orig, obj_orig = optimize_satellites(initial_positions, weights, "original")
    time_orig = time.time() - start_time

    final_distance_orig = great_circle_distance(*optimized_orig[0], *optimized_orig[1])
    final_separation_orig = final_distance_orig * 180 / np.pi

    print(f"  Final separation: {final_separation_orig:.1f}°")
    print(f"  Final positions:")
    print(f"    Sat 1: ({optimized_orig[0][0]:.1f}°, {optimized_orig[0][1]:.1f}°)")
    print(f"    Sat 2: ({optimized_orig[1][0]:.1f}°, {optimized_orig[1][1]:.1f}°)")
    print(f"  Objective: {-obj_orig:.4f}")
    print(f"  Time: {time_orig:.3f}s")

    # Test fixed communication matrix
    print(f"\n2️⃣ FIXED COMMUNICATION MATRIX:")
    start_time = time.time()
    optimized_fixed, obj_fixed = optimize_satellites(initial_positions, weights, "fixed")
    time_fixed = time.time() - start_time

    final_distance_fixed = great_circle_distance(*optimized_fixed[0], *optimized_fixed[1])
    final_separation_fixed = final_distance_fixed * 180 / np.pi

    print(f"  Final separation: {final_separation_fixed:.1f}°")
    print(f"  Final positions:")
    print(f"    Sat 1: ({optimized_fixed[0][0]:.1f}°, {optimized_fixed[0][1]:.1f}°)")
    print(f"    Sat 2: ({optimized_fixed[1][0]:.1f}°, {optimized_fixed[1][1]:.1f}°)")
    print(f"  Objective: {-obj_fixed:.4f}")
    print(f"  Time: {time_fixed:.3f}s")

    # Analysis
    print(f"\n📊 RESULTS ANALYSIS:")
    print(f"  Target: ~45° separation for optimal communication")
    print(f"  Original result: {final_separation_orig:.1f}° (error: {abs(final_separation_orig - 45):.1f}°)")
    print(f"  Fixed result: {final_separation_fixed:.1f}° (error: {abs(final_separation_fixed - 45):.1f}°)")

    if abs(final_separation_fixed - 45) < abs(final_separation_orig - 45):
        print(f"  ✅ IMPROVED! Fixed version is closer to optimal communication distance")
    else:
        print(f"  ⚠️  Still has issues - may need further adjustment")

    # Check if satellites are still collapsing
    if final_separation_fixed < 1.0:
        print(f"  ❌ Satellites still collapsing to same point!")
    elif 40 <= final_separation_fixed <= 50:
        print(f"  ✅ Satellites found good communication distance!")
    else:
        print(f"  ⚠️  Satellites at unexpected separation")

def test_pure_communication():
    """Test pure communication optimization"""
    print(f"\n📡 TESTING PURE COMMUNICATION OPTIMIZATION")
    print("="*50)

    initial = np.array([[0.0, 0.0], [10.0, 10.0]])
    weights = {"efficiency": 0.0, "redundancy": 0.0, "communication": 1.0}

    print(f"Pure communication optimization (should find ~45° separation):")

    for version in ["original", "fixed"]:
        result, obj = optimize_satellites(initial, weights, version)
        final_sep = great_circle_distance(*result[0], *result[1]) * 180 / np.pi
        print(f"  {version:8s}: {final_sep:5.1f}° separation")

if __name__ == "__main__":
    test_communication_matrices()
    test_optimization_with_communication_fix()
    test_pure_communication()