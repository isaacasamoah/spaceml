#!/usr/bin/env python3
"""
Compare scipy, grid search, and random search with fixed constraint functions
"""
import numpy as np
import itertools
import time
from scipy.optimize import minimize

# Import the fixed constraint functions
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

def constraint_matrix_fixed(satellite_positions, constraint_type):
    """Fixed constraint matrix construction (penalty-based)"""
    number_satellites = len(satellite_positions)

    constraint_functions = {
        "efficiency": lambda overlap: overlap,        # Penalty for high overlap
        "redundancy": lambda overlap: 1 - overlap    # Penalty for low overlap
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
                communication_penalty = 1 - communication_quality
                C[i,j] = communication_penalty

    return C

def combined_matrix_fixed(satellite_positions, weights):
    """Combine fixed constraint matrices"""
    efficiency_matrix = constraint_matrix_fixed(satellite_positions, "efficiency")
    redundancy_matrix = constraint_matrix_fixed(satellite_positions, "redundancy")
    signal_matrix = communication_matrix_fixed(satellite_positions)

    C = (weights["efficiency"] * efficiency_matrix +
         weights["redundancy"] * redundancy_matrix +
         weights["communication"] * signal_matrix)

    return C

def objective_fixed(satellite_positions_flat, weights):
    """Calculate eigenvalue objective with fixed constraints"""
    np_positions = np.reshape(satellite_positions_flat, (-1, 2))
    C = combined_matrix_fixed(np_positions, weights)
    eigenvalues = np.linalg.eigvals(C)
    return -np.min(eigenvalues)

def scipy_optimization(initial_positions, weights):
    """Scipy optimization with fixed constraints"""
    flat_positions = initial_positions.flatten()
    bounds = [(-89, 89), (-179, 179)] * len(initial_positions)

    start_time = time.time()
    result = minimize(objective_fixed, flat_positions, args=(weights,),
                     method='L-BFGS-B', bounds=bounds)
    elapsed_time = time.time() - start_time

    optimized_positions = np.reshape(result.x, (-1, 2))
    return optimized_positions, result.fun, elapsed_time, result

def grid_search_optimization(initial_positions, weights, resolution=4):
    """Grid search optimization with fixed constraints"""
    n_satellites = len(initial_positions)

    print(f"🔍 GRID SEARCH OPTIMIZATION")
    print(f"{'='*50}")
    print(f"Satellites: {n_satellites}")
    print(f"Resolution: {resolution} points per coordinate")

    # Create coordinate grid
    lats = np.linspace(-80, 80, resolution)  # Avoid exact poles
    lons = np.linspace(-170, 170, resolution)  # Avoid exact boundaries
    coord_options = [(lat, lon) for lat in lats for lon in lons]

    total_combinations = (resolution * resolution) ** n_satellites
    print(f"Total combinations to test: {total_combinations:,}")

    if total_combinations > 50000:
        print("⚠️  WARNING: This will take a long time!")
        return None, None, None, None

    best_positions = initial_positions.copy()
    best_objective = objective_fixed(initial_positions.flatten(), weights)
    objectives_tested = []

    start_time = time.time()
    tested_count = 0

    print(f"Starting search... (Initial objective: {best_objective:.4f})")

    for coord_assignment in itertools.product(*([coord_options] * n_satellites)):
        test_positions = np.array(coord_assignment)
        test_objective = objective_fixed(test_positions.flatten(), weights)

        objectives_tested.append(test_objective)
        tested_count += 1

        if test_objective < best_objective:
            best_objective = test_objective
            best_positions = test_positions.copy()
            print(f"  New best found! Objective: {best_objective:.4f}")

        if tested_count % 1000 == 0:
            elapsed = time.time() - start_time
            progress = tested_count / total_combinations * 100
            print(f"  Progress: {progress:.1f}% ({tested_count:,}/{total_combinations:,}) - Elapsed: {elapsed:.1f}s")

    elapsed_time = time.time() - start_time

    search_stats = {
        'total_tested': tested_count,
        'best_objective': best_objective,
        'worst_objective': max(objectives_tested) if objectives_tested else None,
        'mean_objective': np.mean(objectives_tested) if objectives_tested else None,
        'std_objective': np.std(objectives_tested) if objectives_tested else None,
        'elapsed_time': elapsed_time
    }

    print(f"\n✅ GRID SEARCH COMPLETE")
    print(f"   Time taken: {elapsed_time:.1f} seconds")
    print(f"   Best objective: {best_objective:.4f}")
    if objectives_tested:
        print(f"   Worst objective: {search_stats['worst_objective']:.4f}")
        print(f"   Mean objective: {search_stats['mean_objective']:.4f}")

    return best_positions, best_objective, elapsed_time, search_stats

def random_search_optimization(initial_positions, weights, n_iterations=5000):
    """Random search optimization with fixed constraints"""
    print(f"🎯 RANDOM SEARCH OPTIMIZATION")
    print(f"{'='*50}")
    print(f"Satellites: {len(initial_positions)}")
    print(f"Iterations: {n_iterations:,}")

    best_positions = initial_positions.copy()
    best_objective = objective_fixed(initial_positions.flatten(), weights)

    objectives_tested = []
    improvement_iterations = []

    start_time = time.time()

    for i in range(n_iterations):
        # Generate random positions
        random_positions = np.column_stack([
            np.random.uniform(-80, 80, len(initial_positions)),    # Avoid exact poles
            np.random.uniform(-170, 170, len(initial_positions))   # Avoid exact boundaries
        ])

        test_objective = objective_fixed(random_positions.flatten(), weights)
        objectives_tested.append(test_objective)

        if test_objective < best_objective:
            best_objective = test_objective
            best_positions = random_positions.copy()
            improvement_iterations.append(i)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Iteration {i+1:,}: Best so far = {best_objective:.4f} (Time: {elapsed:.1f}s)")

    elapsed_time = time.time() - start_time

    search_stats = {
        'total_tested': n_iterations,
        'best_objective': best_objective,
        'worst_objective': max(objectives_tested),
        'mean_objective': np.mean(objectives_tested),
        'std_objective': np.std(objectives_tested),
        'improvement_iterations': improvement_iterations,
        'elapsed_time': elapsed_time
    }

    print(f"\n✅ RANDOM SEARCH COMPLETE")
    print(f"   Time taken: {elapsed_time:.1f} seconds")
    print(f"   Best objective: {best_objective:.4f}")
    print(f"   Improvements found: {len(improvement_iterations)}")

    return best_positions, best_objective, elapsed_time, search_stats

def compare_optimization_methods_fixed(initial_positions, weights, grid_resolution=4, random_iterations=5000):
    """Compare all optimization methods with fixed constraints"""
    print(f"🚀 OPTIMIZATION METHOD COMPARISON (FIXED CONSTRAINTS)")
    print(f"{'='*70}")

    # Calculate initial metrics
    initial_distance = great_circle_distance(*initial_positions[0], *initial_positions[1])
    initial_separation = initial_distance * 180 / np.pi
    initial_obj = objective_fixed(initial_positions.flatten(), weights)

    print(f"Initial configuration:")
    print(f"  Positions: {initial_positions}")
    print(f"  Separation: {initial_separation:.1f}°")
    print(f"  Objective: {initial_obj:.4f}")
    print(f"  Weights: {weights}")

    results = {}

    # 1. Scipy optimization
    print(f"\n1️⃣ SCIPY OPTIMIZATION")
    scipy_positions, scipy_obj, scipy_time, scipy_result = scipy_optimization(initial_positions, weights)
    scipy_separation = great_circle_distance(*scipy_positions[0], *scipy_positions[1]) * 180 / np.pi

    results['scipy'] = {
        'positions': scipy_positions,
        'objective': scipy_obj,
        'separation': scipy_separation,
        'time': scipy_time,
        'success': scipy_result.success
    }
    print(f"   Objective: {scipy_obj:.4f}")
    print(f"   Separation: {scipy_separation:.1f}°")
    print(f"   Success: {scipy_result.success}")
    print(f"   Time: {scipy_time:.3f} seconds")

    # 2. Grid search (if feasible)
    print(f"\n2️⃣ GRID SEARCH OPTIMIZATION")
    if len(initial_positions) <= 2 and grid_resolution <= 5:
        grid_positions, grid_obj, grid_time, grid_stats = grid_search_optimization(
            initial_positions, weights, grid_resolution
        )
        if grid_positions is not None:
            grid_separation = great_circle_distance(*grid_positions[0], *grid_positions[1]) * 180 / np.pi
            results['grid'] = {
                'positions': grid_positions,
                'objective': grid_obj,
                'separation': grid_separation,
                'time': grid_time,
                'stats': grid_stats
            }
    else:
        print("   Skipped - too many combinations for feasible grid search")
        results['grid'] = None

    # 3. Random search
    print(f"\n3️⃣ RANDOM SEARCH OPTIMIZATION")
    random_positions, random_obj, random_time, random_stats = random_search_optimization(
        initial_positions, weights, random_iterations
    )
    random_separation = great_circle_distance(*random_positions[0], *random_positions[1]) * 180 / np.pi

    results['random'] = {
        'positions': random_positions,
        'objective': random_obj,
        'separation': random_separation,
        'time': random_time,
        'stats': random_stats
    }

    # Summary comparison
    print(f"\n📊 RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Objective':<12} {'Separation':<12} {'Time (s)':<10} {'Status'}")
    print(f"{'-'*80}")

    # Find best objective
    best_obj = min([results[k]['objective'] for k in results if results[k] is not None])

    print(f"{'Scipy':<15} {results['scipy']['objective']:<12.4f} {results['scipy']['separation']:<12.1f}° {results['scipy']['time']:<10.3f} {'🏆' if results['scipy']['objective'] == best_obj else ''}")

    if results['grid'] is not None:
        status = '🏆' if results['grid']['objective'] == best_obj else ''
        print(f"{'Grid Search':<15} {results['grid']['objective']:<12.4f} {results['grid']['separation']:<12.1f}° {results['grid']['time']:<10.3f} {status}")

    status = '🏆' if results['random']['objective'] == best_obj else ''
    print(f"{'Random Search':<15} {results['random']['objective']:<12.4f} {results['random']['separation']:<12.1f}° {results['random']['time']:<10.3f} {status}")

    # Analysis
    print(f"\n🔍 ANALYSIS:")
    obj_diffs = []
    if results['grid'] is not None:
        diff = abs(results['scipy']['objective'] - results['grid']['objective'])
        obj_diffs.append(diff)
        print(f"   Scipy vs Grid difference: {diff:.6f}")

    diff = abs(results['scipy']['objective'] - results['random']['objective'])
    obj_diffs.append(diff)
    print(f"   Scipy vs Random difference: {diff:.6f}")

    max_diff = max(obj_diffs) if obj_diffs else 0
    if max_diff < 0.001:
        print(f"   ✅ All methods agree well - scipy likely finding global optimum")
    elif max_diff < 0.01:
        print(f"   ⚠️  Small differences - scipy may be in local minimum")
    else:
        print(f"   ❌ Large differences - scipy likely stuck in local minimum")

    return results

def test_different_scenarios():
    """Test optimization comparison on different scenarios"""
    print(f"\n🧪 TESTING DIFFERENT OPTIMIZATION SCENARIOS")
    print(f"{'='*60}")

    scenarios = [
        {
            'name': 'Efficiency Heavy',
            'initial': np.array([[0.0, 0.0], [10.0, 10.0]]),
            'weights': {"efficiency": 0.8, "redundancy": 0.1, "communication": 0.1},
            'expected': 'Large separation (~180°)'
        },
        {
            'name': 'Communication Heavy',
            'initial': np.array([[0.0, 0.0], [10.0, 10.0]]),
            'weights': {"efficiency": 0.1, "redundancy": 0.1, "communication": 0.8},
            'expected': 'Medium separation (~45°)'
        },
        {
            'name': 'Redundancy Heavy',
            'initial': np.array([[0.0, 0.0], [60.0, 60.0]]),
            'weights': {"efficiency": 0.1, "redundancy": 0.8, "communication": 0.1},
            'expected': 'Small separation (close together)'
        }
    ]

    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Expected: {scenario['expected']}")
        print(f"   Weights: {scenario['weights']}")

        results = compare_optimization_methods_fixed(
            scenario['initial'],
            scenario['weights'],
            grid_resolution=3,  # Smaller for speed
            random_iterations=2000
        )

        print(f"   Result: Scipy found {results['scipy']['separation']:.1f}° separation")

if __name__ == "__main__":
    test_different_scenarios()