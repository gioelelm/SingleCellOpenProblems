# Copyright (C) 2020 Krishnaswamy Lab, Yale University

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn


def _preprocess(adata):
    """Library size normalize and sqrt transform adata"""
    result_dict = sc.pp.normalize_total(adata, target_sum=1e5, inplace=False)
    adata.layers["X_norm"] = result_dict["X"]
    adata.obs["norm_factor"] = result_dict["norm_factor"]
    adata.layers["X_norm_sqrt"] = adata.layers["X_norm"].sqrt()

    # Do PCA
    adata.obsm["X_pca"] = sc.pp.pca(adata.layers["X_norm_sqrt"])
    adata.uns["preprocessed_for_differential_abundance"] = True
    return adata


def _diffuse_values(adata, thresh=1e-6, max_iter=10000, ref_neighbors=7):
    """Uses diffusion to interpolate between fixed values at set reference points

    Parameters
    ----------
    adata : AnnData
        AnnData object must have the following keys in `uns`: 'reference_index',
        'reference_values', 'graph' OR .obsp 'connectivities'
    thresh : Float
        Stopping condition for diffusion. Each iteration, the difference between the
        diffused values from the previous and current step is calculated. Once the delta
        is less than `thresh`, the process stops
    max_iter : Int
        Maximum number of iterations before diffusion is stopped.
    ref_neighbors : Int
        Number of neighbors used to expand reference values

    Returns
    -------
    values_interp
        Values diffused over the graph from the reference points.

    """

    # Get reference indicies
    reference_index = adata.uns["reference_index"]
    n_ref = len(reference_index)

    # Mask to get values that aren't references for interpolation
    not_reference = ~np.isin(np.arange(adata.n_obs), reference_index)

    # Generate signal values at each reference
    # Assign values to peaks
    values = np.tile([0, 1], n_ref // 2 + 1)[:n_ref]
    np.random.shuffle(values)
    adata.uns["reference_values"] = values

    # Get graph
    try:
        graph = adata.uns["graph"]
    except AttributeError:
        graph = gt.Graph(adata.obsp["connectivities"], precomputed="adjacency")

    # Expand reference points to kNN neighborhood centered at each reference index
    kneighbors = graph._knn_tree.kneighbors(
        adata.obsm["X_pca"][reference_index],
        n_neighbors=ref_neighbors,
        return_distance=False,
    )

    # Collect new indices and values for extended reference set including neighbors of
    # each reference
    ref_index_extended = np.array([], int)
    values_extended = np.array([], float)
    for i, ref_ix in enumerate(reference_index):
        curr_value = values[i]  # Select the reference value for the original point
        curr_neighbors = kneighbors[i]  # Get the NN for the original point
        curr_values_ext = np.full(curr_neighbors.shape[0], curr_value)
        # Add original reference to extended array
        ref_index_extended = np.hstack((ref_index_extended, np.array(ref_ix)))
        values_extended = np.hstack((values_extended, np.array(curr_value)))

        # Add neighbors to extended array
        ref_index_extended = np.hstack((ref_index_extended, curr_neighbors))
        values_extended = np.hstack((values_extended, curr_values_ext))

    adata.uns["reference_index_ext"] = ref_index_extended
    adata.uns["reference_values_ext"] = values_extended

    # Iteratively diffuse values across the graph
    values_interp = np.full(adata.n_obs, 0.5)
    i = -1
    cond = True
    while cond:
        i += 1
        # Do diffusion step
        values_interp[ref_index_extended] = values_extended
        values_interp_new = graph.P @ values_interp

        if n_iter == None:
            # In this case, we're stopping based on thresh
            diff = np.sum((values_interp - values_interp_new) ** 2)
            cond = diff > thresh

            # Also check for max_iter
            if i > max_iter:
                cond = False
        # Always update the values
        values_interp = values_interp_new

    return values_interp


def _create_pdf_from_embedding(data_embedding):
    # DEPRECATED 3/1/21 - This is outdated method

    # Given a data_embedding, sample a simplex to weight each dimension to get a
    # new PDF for each condition

    # Create an array of values that sums to 1
    n_components = data_embedding.shape[1]
    data_simplex = np.sort(np.random.uniform(size=(n_components - 1)))
    data_simplex = np.hstack([0, data_simplex, 1])
    data_simplex = np.diff(data_simplex)
    np.random.shuffle(data_simplex)  # operates inplace

    # Weight each embedding component by the simplex weights
    sort_axis = np.sum(data_embedding * data_simplex, axis=1)

    # Pass the weighted components through a logit
    pdf = scipy.special.expit(sort_axis)
    if np.random.choice([True, False]):
        pdf = 1 - pdf
    return pdf


def _create_pdf_from_graph(graph, n_peaks):
    return


def simulate_treatment(
    adata: sc.AnnData,
    n_conditions: int = 2,
    n_replicates: int = 2,
    embedding_name: str = "X_pca",
    n_components: int = 10,
    seed: [int, np.random.RandomState] = None,
) -> np.ndarray:
    """Creates random differentially expressed regions over a dataset for benchmarking.

    Parameters
    ----------
    seed : integer or numpy.RandomState, optional, default: None
        Random state. Defaults to the global `numpy` random number generator
    adata : AnnData
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    n_conditions : int, optional, default: 2
        Number of conditions to simulate
    n_replicates : int, optional, default: 2
        Number of replicates to simulate
    embedding_name : str, optional, default: "X_pca"
        Name of embedding in adata.obsm to use to simulate differential abundance
    n_components : int, optional, default: 10
        Number of dimensions of adata.obsm['embedding_name'] to use for simulation.
        For embeddings sorted by explained variance, like PCA, more components results
        in more uniform enrichment / depletion
    seed : [int, np.RandomState], default: None


    Returns
    ----------
    condition : array-like, shape=[n_obs,]
        Condition assiment for each cell
    replicate : array-like, shape=[n_obs,]
        Replicate assiment for each cell
    condition_probability : pandas.DataFrame, shape=[n_obs, n_conditions]
        DataFrame with the corresponding probabiltiy for each condition
    """

    np.random.seed(seed)
    if "preprocessed_for_differential_abundance" not in adata.uns:
        _preprocess(adata)

    data_embedding = adata.obsm[embedding_name]
    if not np.isclose(data_embedding.mean(), 0):
        # embedding data must be mean-centered
        data_embedding = scipy.stats.zscore(data_embedding, axis=0)

    # Randomly flip sign of each embedding dimension
    data_embedding *= np.random.choice([-1, 1], size=data_embedding.shape[1])

    # Create information about each condition and replicate
    conditions = ["condition{}".format(i) for i in range(1, n_conditions + 1)]
    replicates = ["replicate{}".format(i) for i in range(1, n_replicates + 1)]

    # Create one PDF for each condition
    condition_probability = []
    for condition in conditions:
        pdf = _create_pdf(data_embedding[:, :n_components]).reshape(-1, 1)
        condition_probability.append(pdf)
    condition_probability = np.concatenate(condition_probability, axis=1)

    # Normalize PDF for each condition to sum to 1
    condition_probability = sklearn.preprocessing.normalize(
        condition_probability,
        norm="l1",
    )
    condition_probability = pd.DataFrame(
        condition_probability,
        columns=conditions,
        index=adata.obs_names,
    )

    condition = []
    for ix, prob in condition_probability.iterrows():
        condition.append(np.random.choice(condition_probability.columns, p=prob))

    replicate = np.random.choice(replicates, size=adata.n_obs)

    # Assign attributes to adata
    adata.obs["condition"] = condition
    adata.obs["replicate"] = replicate
    adata.obs["sample"] = [
        ".".join(row) for _, row in adata.obs[["condition", "replicate"]].iterrows()
    ]
    adata.obsm["ground_truth_probability"] = condition_probability
    adata.uns["conditions"] = conditions
    adata.uns["replicates"] = replicates
    adata.uns["samples"] = np.unique(adata.obs["sample"])
    adata.uns["n_conditions"] = n_conditions
    adata.uns["n_replicates"] = n_replicates
    adata.uns["n_samples"] = n_conditions * n_replicates

    return adata
