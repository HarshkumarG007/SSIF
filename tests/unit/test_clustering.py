"""
test_clustering.py — Unit tests for trajectory phenotype clustering.
Student Success Intelligence Framework (SSIF)
"""
import pytest
from src.retention.clustering import (
    CLUSTER_FEATURES,
    TrajectoryClusteringResult,
    run_trajectory_clustering,
)


def test_trajectory_clustering_execution():
    # Run with small bootstrap sample for fast unit test execution
    res = run_trajectory_clustering(n_clusters=3, n_bootstrap=3, random_state=42)
    assert isinstance(res, TrajectoryClusteringResult)
    assert res.optimal_k == 3
    assert res.n_students > 0
    assert -1.0 <= res.bootstrap_mean_ari <= 1.0
    
    # Check cluster profiles dataframe
    profiles = res.cluster_profiles
    assert len(profiles) == 3
    assert "Cluster_ID" in profiles.columns
    assert "Phenotype" in profiles.columns
    assert "N_Students" in profiles.columns
    assert "Eventual_Dropout_Rate_Pct" in profiles.columns
    assert profiles["N_Students"].sum() == res.n_students
