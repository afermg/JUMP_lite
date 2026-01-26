"""Feature selection operations for morphological profiles."""

import numpy as np
import polars as pl


def variance_threshold(
    df: pl.DataFrame,
    features: list[str],
    freq_cut: float = 0.05,
    unique_cut: float = 0.01,
    var_threshold: float | None = None,
) -> list[str]:
    """
    Remove low-variance features.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        freq_cut: Ratio of 2nd to most common value
        unique_cut: Ratio of unique values to samples
        var_threshold: Minimum variance required (None = skip variance check)

    Returns:
        List of features to keep
    """
    if not features or len(df) == 0:
        return features

    X = df.select(features).to_numpy()
    n_samples = len(X)

    keep_features = []

    count_variance_drop = 0
    count_unique_drop = 0
    count_freq_drop = 0

    for i, feat in enumerate(features):
        col = X[:, i]

        # Actual variance check (if enabled)
        if var_threshold is not None:
            # Use nanvar to safely handle NaN and inf values
            variance = np.nanvar(col)
            if not np.isfinite(variance) or variance < var_threshold:
                count_variance_drop += 1
                continue

        # Clean column for unique/frequency checks
        col_clean = col[~np.isnan(col) & np.isfinite(col)]

        if len(col_clean) == 0:
            count_variance_drop += 1
            continue

        # Count unique values
        unique_values, counts = np.unique(col_clean, return_counts=True)
        n_unique = len(unique_values)

        # Skip if too few unique values
        if n_unique / n_samples < unique_cut:
            count_unique_drop += 1
            continue

        # Check frequency ratio
        if len(counts) >= 2:
            sorted_counts = np.sort(counts)[::-1]
            freq_ratio = sorted_counts[1] / sorted_counts[0] if sorted_counts[0] > 0 else 0
            if freq_ratio < freq_cut:
                count_freq_drop += 1
                continue

        keep_features.append(feat)

    print(f"    Variance threshold: dropped {count_variance_drop} features")
    print(f"    Unique value threshold: dropped {count_unique_drop} features")
    print(f"    Frequency ratio threshold: dropped {count_freq_drop} features")

    return keep_features


def correlation_threshold(
    df: pl.DataFrame,
    features: list[str],
    threshold: float = 0.9,
    method: str = "pearson",
) -> list[str]:
    """
    Remove highly correlated features using greedy independent set.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        threshold: Correlation cutoff
        method: Correlation method (pearson, spearman, kendall)

    Returns:
        List of features to keep
    """
    X = df.select(features).to_numpy()

    # Compute correlation matrix
    if method == "pearson":
        corr = np.corrcoef(X, rowvar=False)
    elif method == "spearman":
        from scipy.stats import spearmanr
        corr, _ = spearmanr(X, axis=0)
    elif method == "kendall":
        from scipy.stats import kendalltau
        n_features = len(features)
        corr = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i, n_features):
                tau, _ = kendalltau(X[:, i], X[:, j])
                corr[i, j] = corr[j, i] = tau
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Handle NaN and inf in correlation matrix
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Create adjacency graph
    adj_graph = np.abs(corr) > threshold

    # Find redundant features using greedy independent set
    redundant_indices = greedy_independent_set(adj_graph)

    # Return features to keep
    keep_features = [feat for i, feat in enumerate(features) if i not in redundant_indices]

    return keep_features


def drop_na_columns(
    df: pl.DataFrame, features: list[str], na_cutoff: float = 0.05
) -> list[str]:
    """
    Remove features with too many missing or infinite values.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        na_cutoff: Maximum fraction of NaN/inf values allowed

    Returns:
        List of features to keep
    """
    n_samples = len(df)
    keep_features = []
    dropped_features = []
    
    
    
    # import pdb; pdb.set_trace()
    for feat in features:
        # Count both null and infinite values
        n_null = df[feat].null_count()
        n_nan = df[feat].is_nan().sum()
        n_inf = df[feat].is_infinite().sum()
        n_invalid = n_null + n_nan + n_inf
        
        if n_invalid / n_samples <= na_cutoff:
            keep_features.append(feat)
        else:
            dropped_features.append(feat)

    return keep_features, dropped_features


def drop_outliers(
    df: pl.DataFrame, features: list[str], outlier_cutoff: float = 500
) -> list[str]:
    """
    Remove features with extreme values using z-score normalization.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        outlier_cutoff: Maximum absolute z-score allowed (default: 500)

    Returns:
        List of features to keep
    """
    if not features or len(df) == 0:
        return features

    X = df.select(features).to_numpy()

    # Z-normalize features using NaN-safe operations
    # (add epsilon to denominator to prevent division by zero)
    X_z_norm = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)

    # Compute max/min absolute z-scores (ignoring NaN/inf)
    max_vals = np.abs(np.nanmax(X_z_norm, axis=0))
    min_vals = np.abs(np.nanmin(X_z_norm, axis=0))

    # Handle features where max/min are non-finite (all NaN, or numerical issues)
    max_vals = np.nan_to_num(max_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)
    min_vals = np.nan_to_num(min_vals, nan=np.inf, posinf=np.inf, neginf=np.inf)

    vals_within_cutoff = (max_vals <= outlier_cutoff) & (min_vals <= outlier_cutoff)
    keep_indices = np.where(vals_within_cutoff)[0]

    keep_features = [features[i] for i in keep_indices]

    return keep_features


def blocklist_filter(
    df: pl.DataFrame, features: list[str], blocklist: list[str] | None = None
) -> list[str]:
    """
    Remove blocklisted features.

    Args:
        df: Input DataFrame
        features: Feature columns to evaluate
        blocklist: List of feature patterns to exclude (None = use default)

    Returns:
        List of features to keep
    """
    if blocklist is None:
        # Default blocklist: common problematic features
        blocklist = [
            "Location_Center",
            "Location_Max",
            "Location_Min",
            "Number_Object_Number",
            "ExecutionTime",
        ]

    keep_features = []

    for feat in features:
        # Check if feature matches any blocklist pattern
        is_blocked = any(pattern in feat for pattern in blocklist)
        if not is_blocked:
            keep_features.append(feat)

    return keep_features


def greedy_independent_set(adj_graph: np.ndarray) -> list[int]:
    """
    Find redundant features using greedy independent set algorithm.

    Args:
        adj_graph: Binary adjacency matrix

    Returns:
        Indices of features to remove
    """
    adj_graph = adj_graph.copy()
    np.fill_diagonal(adj_graph, 0)

    remaining_nodes = set(range(adj_graph.shape[0]))
    independent_set = []

    while remaining_nodes:
        degrees = np.sum(adj_graph, axis=1)
        min_degree_node = min(remaining_nodes, key=lambda node: degrees[node])

        independent_set.append(min_degree_node)

        neighbors = set(np.where(adj_graph[min_degree_node] == 1)[0])
        remaining_nodes -= neighbors | {min_degree_node}

        adj_graph[min_degree_node, :] = 0
        adj_graph[:, min_degree_node] = 0
        for neighbor in neighbors:
            adj_graph[neighbor, :] = 0
            adj_graph[:, neighbor] = 0

    return list(set(np.arange(len(adj_graph))).difference(independent_set))


def select_features(
    df: pl.DataFrame,
    features: list[str],
    operations: list[dict],
    verbose: bool = True,
) -> list[str]:
    """
    Apply sequential feature selection operations.

    Args:
        df: Input DataFrame
        features: Starting feature list
        operations: List of operation configs
        verbose: Print excluded feature counts

    Returns:
        Selected feature names
    """
    current_features = features.copy()

    for op in operations:
        name = op["name"]
        params = {k: v for k, v in op.items() if k != "name"}

        if name == "variance_threshold":
            new_features = variance_threshold(df, current_features, **params)
        elif name == "correlation_threshold":
            new_features = correlation_threshold(df, current_features, **params)
        elif name == "drop_na_columns":
            new_features = drop_na_columns(df, current_features, **params)
        elif name == "drop_outliers":
            new_features = drop_outliers(df, current_features, **params)
        elif name == "blocklist":
            new_features = blocklist_filter(df, current_features, **params)
        else:
            raise ValueError(f"Unknown selection operation: {name}")

        if verbose:
            n_dropped = len(current_features) - len(new_features)
            print(f"  {name}: dropped {n_dropped} features ({len(new_features)} remaining)")

        # Safeguard: if operation would remove all features, skip it
        if len(new_features) == 0 and len(current_features) > 0:
            print(f"  WARNING: {name} would remove all features, skipping this filter")
            continue

        current_features = new_features

    return current_features
