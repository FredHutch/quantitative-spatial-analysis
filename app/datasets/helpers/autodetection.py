from app.models.points import SpatialPoints
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from string import ascii_uppercase
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _rotate_points(coords: pd.DataFrame, angle: float, cnames=['x', 'y']):
    r = Rotation.from_euler('z', angle, degrees=True)
    rot_coords = pd.DataFrame(
        r.apply(
            coords
            .reindex(columns=cnames)
            .assign(z=0)
            .values
        ),
        columns=cnames + ["z"],
        index=coords.index
    ).reindex(columns=cnames)
    return rot_coords


def _rotate_cores(cores: pd.DataFrame, angle: float):
    r = Rotation.from_euler('z', angle, degrees=True)
    return pd.concat([
        pd.DataFrame(
            r.apply(
                cores
                .reindex(columns=['x', 'y'])
                .assign(z=0)
                .values
            ),
            columns=['x', 'y', 'z'],
            index=cores.index
        ).drop(columns=['z']),
        cores.drop(columns=['x', 'y'])
    ], axis=1)


def _find_grid(vals: pd.Series, max_n=16, min_n=2):
    """Use k-means clustering to find the points with maximal density."""

    # Test different numbers of grid lines
    k_stats = {
        k: _run_gaussian_mixture(vals, k)
        for k in range(min_n, max_n+1)
    }

    # Pick the top value of k
    best_model, best_score = None, None

    for model, score in k_stats.values():
        if best_score is None or score > best_score:
            best_model, best_score = model, score

    return np.sort(best_model.means_[:, 0])


def _run_gaussian_mixture(vals: pd.Series, k: int):

    # Make a matrix of values
    X = pd.DataFrame(dict(x=vals)).values

    # Fit the model
    gm = GaussianMixture(n_components=k, random_state=0).fit(X)

    # Predict labels for each point
    pred = gm.predict(X)

    # Get the probability for each point
    proba = [
        i_prob[i]
        for i, i_prob in zip(pred, gm.predict_proba(X))
    ]

    # Return the model, and also the mean probability
    return gm, np.mean(proba)


def _find_single_core(coords: pd.DataFrame, x: float, y: float, radius: float, tol=0.001):

    # Get all the points inside the bounding box
    in_box = (
        coords
        .query(f"x > {x - radius}")
        .query(f"x < {x + radius}")
        .query(f"y > {y - radius}")
        .query(f"y < {y + radius}")
    )

    # Get the mean x/y values
    mean_x = in_box["x"].mean()
    mean_y = in_box["y"].mean()

    # If the new point is > tol away from the starting point
    if (
        (np.abs(x - mean_x) / x) > tol
        or
        (np.abs(y - mean_y) / y) > tol
    ):
        return _find_single_core(coords, mean_x, mean_y, radius, tol=tol)
    else:
        return dict(
            x=mean_x, y=mean_y, radius=radius
        )


def _shrink_core_size(coords: pd.DataFrame, core: dict, radius: float, q=0.99, r=1.05):

    if coords.shape[0] == 0:
        return coords

    # 1. Get the points inside this core
    # 2. Calculate the distance from the center
    # 3. Get the distance that includes 99% of points
    # 4. Count the number of points inside that circle

    core_points = (
        coords
        .query(f"x > {core['x'] - radius}")
        .query(f"x < {core['x'] + radius}")
        .query(f"y > {core['y'] - radius}")
        .query(f"y < {core['y'] + radius}")
    )

    dists = distance.cdist(core_points[["x", "y"]], [[core['x'], core['y']]])
    radius = np.quantile(dists[:, 0], q) * r

    n = int(np.sum(dists <= radius))

    core['radius'] = radius
    core['n'] = n


def _find_cores(
    coords: pd.DataFrame,
    x_grid: list,
    y_grid: list,
    min_prop_cells=0.001
) -> pd.DataFrame:
    """
    Find the individual cores.

    1. Using a large radius, find the center of every core by iteratively finding the
    centroid of all points inside the grid.
    2. Test a range of radius values which cover the largest number of points.
    3. Only keep non-overlapping cores.
    """
    # Set the radius as the median offset between grid points
    radius = np.mean([np.median(grid[1:] - grid[:-1]) / 2 for grid in [x_grid, y_grid]])

    # For each point in the grid, iteratively find the best centroid
    cores = [
        {
            "col_i": col_i,
            "row_i": row_i,
            **_find_single_core(coords, x, y, radius)
        }
        for col_i, x in enumerate(x_grid)
        for row_i, y in enumerate(y_grid)
    ]

    # Find the circle size for each core,
    # adjust size based on the number of cells
    # and count the number of cells in each one
    for core in cores:
        _shrink_core_size(coords, core, radius)

    min_n_cells = min_prop_cells * coords.shape[0]
    n_cores_all = len(cores)
    cores = [core for core in cores if core['n'] > min_n_cells]
    n_cores_enough_cells = len(cores)

    # Sort by the number of cells
    cores.sort(key=lambda i: i['n'], reverse=True)

    # Get the pairwise distances between cores
    core_pdist = distance.squareform(
        distance.pdist([
            [core['x'], core['y']]
            for core in cores
        ])
    )
    touching = np.array([
        [
            core_pdist[i, j] < (core_i['radius'] + core_j['radius'])
            for j, core_j in enumerate(cores)
        ]
        for i, core_i in enumerate(cores)
    ])
    cores = [
        core for i, core in enumerate(cores)
        if i == 0 or not any(touching[i, :i])
    ]
    n_cores_not_touching = len(cores)

    logger.info(f"All Cores: {n_cores_all:,}")
    logger.info(f"With enough cells: {n_cores_enough_cells:,}")
    logger.info(f"Not touching: {n_cores_not_touching:,}")

    return pd.DataFrame(cores)


def find_tma_cores(
    points: SpatialPoints,
    angle: float,
    subsample_n=10000,
    min_prop_cells=0.001
):

    # Rotate the points if requested
    coords = _rotate_points(
        points.coords.rename(
            columns={
                points.xcol: "x",
                points.ycol: "y"
            }
        ),
        angle=angle
    )

    # Pick the grid lines for each axis
    x_grid = _find_grid(coords["x"].sample(subsample_n))
    y_grid = _find_grid(coords["y"].sample(subsample_n))

    cores = _find_cores(coords, x_grid, y_grid, min_prop_cells=min_prop_cells)

    # Rotate the cores back
    cores = _rotate_cores(cores, -angle)

    return cores


def name_tma_cores(
    cores: pd.DataFrame,
    core_naming_scheme: str,
    row_start: str,
    col_start: str
):

    rows_are_letters = core_naming_scheme == "Row=Letter; Column=Number"
    if not rows_are_letters:
        if core_naming_scheme != "Column=Letter; Row=Number":
            raise ValueError(f"Unexpected core naming scheme: {core_naming_scheme}")

    row_map = _make_index_map(
        cores['row_i'],
        row_start == "Bottom",
        rows_are_letters
    )

    col_map = _make_index_map(
        cores['col_i'],
        col_start == "Left",
        not rows_are_letters
    )

    return cores.assign(
        name=cores.apply(
            lambda r: f"{row_map[r['row_i']]}{col_map[r['col_i']]}",
            axis=1
        )
    )


def _make_index_map(vals: pd.Series, ascending: bool, are_letters: bool):
    return dict(zip(
        (
            vals
            .drop_duplicates()
            .sort_values(
                ascending=ascending
            )
            .tolist()
        ),
        (
            ascii_uppercase
            if are_letters
            else range(1, 1+vals.shape[0])
        )
    ))

