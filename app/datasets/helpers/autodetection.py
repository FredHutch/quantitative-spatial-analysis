from scipy.spatial import ConvexHull
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


def _rotate_cores(cores: dict, angle: float):
    r = Rotation.from_euler('z', angle, degrees=True)
    for core in cores:
        core["shape"] = r.apply(
            np.hstack([
                core["shape"],
                np.zeros((core["shape"].shape[0], 1))
            ])
        )[:, :2]


def guess_tma_grid(points: SpatialPoints, angle: float = 0.0, subsample_n=100000):
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

    ncols = _guess_n(coords["x"].sample(subsample_n))
    nrows = _guess_n(coords["y"].sample(subsample_n))
    return nrows, ncols


def _guess_n(vals: pd.Series, max_n=16, min_n=2):
    """Use k-means clustering to guess the number of rows/columns."""
    # Use k-means clustering to find the number of rows/columns
    k_stats = {
        k: _run_gaussian_mixture(vals, k)
        for k in range(min_n, max_n + 1)
    }

    # Pick the top value of k
    best_score = None
    best_n = None

    for n, (model, score) in k_stats.items():
        if best_score is None or score > best_score:
            best_score, best_n = score, n

    return best_n


def _find_grid(vals: pd.Series, n: int):
    """Use k-means clustering to find the points with maximal density."""

    # Run the mixture model
    model, _ = _run_gaussian_mixture(vals, n)

    # Get the coordinates of the grid lines
    grid = np.sort(model.means_[:, 0])

    # Find the median distance between the grid lines
    dists = np.diff(grid, n=1)
    median_dist = np.median(dists)

    # If there are any points which are > 1.75X the median distance,
    # add a point in between them
    for i in range(1, len(grid)):
        if dists[i - 1] > 1.75 * median_dist:
            new_point = (grid[i - 1] + grid[i]) / 2
            grid = np.insert(grid, i, new_point)

    return grid


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


def _find_centroid_from_points(
    coords: pd.DataFrame,
    x: float,
    y: float,
    radius: float
):
    """
    Find the centroid of points within a given radius.
    Ignore points which are within any existing core.
    """
    in_box = (
        coords
        .query(f"x > {x - radius}")
        .query(f"x < {x + radius}")
        .query(f"y > {y - radius}")
        .query(f"y < {y + radius}")
    )

    if in_box.shape[0] == 0:
        return None, None

    mean_x = in_box["x"].mean()
    mean_y = in_box["y"].mean()

    return mean_x, mean_y


def _drop_points_in_cores(cores: list, coords: pd.DataFrame):
    """
    Drop points which are within any existing core.
    """
    if len(cores) == 0:
        return coords

    for core in cores:
        # If the core has no radius, skip it
        if core['radius'] <= 0:
            continue

        # Drop points which are within this core
        coords = coords.query(
            f"x < {core['x'] - core['radius']} or "
            f"x > {core['x'] + core['radius']} or "
            f"y < {core['y'] - core['radius']} or "
            f"y > {core['y'] + core['radius']}"
        )

    return coords


def _find_single_core(
    cores: list,
    coords: pd.DataFrame,
    x: float,
    y: float,
    radius: float,
    col_i: int,
    row_i: int,
    tol=0.001,
    max_iter=10
):
    """Append a single core to the list of cores."""
    logger.info(f"Finding core at ({x}, {y}) with radius {radius}.")

    # # Get the points which are not in any existing core
    # coords = _drop_points_in_cores(cores, coords)

    # Get the centroid of the points within the radius
    mean_x, mean_y = _find_centroid_from_points(coords, x, y, radius)

    if mean_x is None or mean_y is None:
        logger.info(f"No points found in the box around ({x}, {y}) with radius {radius}.")
        return

    # If the new point is > tol away from the starting point, move the x/y
    iter_n = 0
    while (
        (np.abs(x - mean_x) / x) > tol
        or
        (np.abs(y - mean_y) / y) > tol
    ):
        iter_n += 1
        if iter_n > max_iter:
            break
        x, y = mean_x, mean_y
        mean_x, mean_y = _find_centroid_from_points(coords, mean_x, mean_y, radius)
        break

    if mean_x is None or mean_y is None:
        logger.info(f"No points found in the box around ({x}, {y}) with radius {radius}.")
        return

    # Add the new core
    cores.append({
        'x': mean_x,
        'y': mean_y,
        'col_i': col_i,
        'row_i': row_i,
        'radius': radius,
        'n': 0  # This will be updated later
    })

    # Shrink the radius to include 99% of points
    _shrink_core_size(coords, cores[-1], radius)
    logger.info(f"Found core at ({mean_x}, {mean_y}) with radius {radius}.")


def _shrink_core_size(coords: pd.DataFrame, core: dict, radius: float, q=0.99, r=1.05):

    if coords.shape[0] == 0:
        return coords

    # 1. Get the points inside this core
    # 2. Calculate the distance from the center
    # 3. Get the distance that includes 99% of points
    # 4. Count the number of points inside that circle

    logger.info(core)
    logger.info(radius)
    logger.info(coords.shape)
    core_points = (
        coords
        .query(f"x >= {core['x'] - radius}")
        .query(f"x <= {core['x'] + radius}")
        .query(f"y >= {core['y'] - radius}")
        .query(f"y <= {core['y'] + radius}")
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
    min_prop_cells=0.001,
    minor_grid_scale=20,
    n_iter=50
) -> pd.DataFrame:
    """
    Find the individual cores.

    1. Set the initial positions of the core centroids using the provided major grid lines.
    2. For each cell, round off the coordinates to the nearest minor grid line
       (The minor grid scale is the number of lines between each major grid line.)
    3. Group each occupied minor grid cell according to which core centroid it is closest to.
    4. Iteratively update the core centroid positions based on the points assigned to each core.
    """

    # Use the major grid lines to set the initial positions of the cores
    cores = pd.DataFrame([
        dict(
            x=x,
            y=y,
            col_i=col_i,
            row_i=row_i,
            id=f"core_{col_i}_{row_i}",
        )
        for row_i, x in enumerate(x_grid)
        for col_i, y in enumerate(y_grid)
    ]).set_index("id")

    # Find the median distance beween the major grid lines
    x_dists = np.diff(x_grid, n=1)
    y_dists = np.diff(y_grid, n=1)
    x_median_dist = np.median(x_dists)
    y_median_dist = np.median(y_dists)

    # To snap all points to the nearest minor grid line,
    # we need to find the minor grid size
    minor_grid_size = min(x_median_dist, y_median_dist) / minor_grid_scale

    # Round the coordinates to the nearest minor grid line
    coords = coords.assign(
        grid_x=lambda df: (df['x'] / minor_grid_size).apply(int),
        grid_y=lambda df: (df['y'] / minor_grid_size).apply(int)
    )
    cores = cores.assign(
        grid_x=lambda df: (df['x'] / minor_grid_size).apply(int),
        grid_y=lambda df: (df['y'] / minor_grid_size).apply(int)
    )

    # Remove any points that are outside the grid
    coords = coords.query(
        f"x >= {x_grid[0] - (2 * x_median_dist)} and "
        f"x <= {x_grid[-1] + (2 * x_median_dist)} and "
        f"y >= {y_grid[0] - (2 * y_median_dist)} and "
        f"y <= {y_grid[-1] + (2 * y_median_dist)}"
    )

    # Get the unique minor grid cells
    minor_grid_cells = coords[['grid_x', 'grid_y']].drop_duplicates()

    for _ in range(n_iter):
        logger.info(f"Iteration {_+1}/{n_iter}")

        # Annotate the minor grid cells with the closest core
        minor_grid_cells = _find_closest_core(minor_grid_cells, cores)

        # Find the centroid for each core, based on the collection of minor grid cells
        # annotated as being closest to that core
        cores = _find_core_centroids(minor_grid_cells)

    # Annotate the original points with the closest core
    coords = coords.merge(
        minor_grid_cells,
        on=['grid_x', 'grid_y'],
        how='left',
    )

    # For each core, draw a hull around the points assigned to it
    hulls = {
        core_id: _draw_hull(core_coords)
        for core_id, core_coords in coords.groupby('closest_core')
        if core_coords.shape[0] >= 3
    }

    # Find the minimum number of cells required for a core
    min_n_cells = int(min_prop_cells * coords.shape[0])

    # Count the number of cells in each core
    core_size = coords.groupby('closest_core').size()

    return [
        {
            'core_id': core_id,
            'row_i': int(core_id.split('_')[1]),
            'col_i': int(core_id.split('_')[2]),
            'n': core_size[core_id],
            'shape': hulls[core_id],
        }
        for core_id in core_size.index
        if core_size.get(core_id, 0) >= min_n_cells
    ]


def _find_closest_core(minor_grid_cells: pd.DataFrame, cores: pd.DataFrame):
    """
    For each minor grid cell, find the closest core.
    """
    # Calculate the distance from each minor grid cell to each core
    dists = distance.cdist(
        minor_grid_cells[['grid_x', 'grid_y']].values,
        cores[['grid_x', 'grid_y']].values
    )

    # Find the index of the closest core for each minor grid cell
    closest_core_idx = np.argmin(dists, axis=1)

    # Find the distance to the closest core
    closest_core_dist = dists[np.arange(dists.shape[0]), closest_core_idx]

    # Assign the closest core index to each minor grid cell
    minor_grid_cells['closest_core'] = [
        cores.index[i] for i in closest_core_idx
    ]
    minor_grid_cells['closest_dist'] = closest_core_dist

    return minor_grid_cells


def _find_core_centroids(minor_grid_cells: pd.DataFrame):
    """
    For each core, find the centroid based on the minor grid cells assigned to it.
    Only use the cells which are within 2X the median distance to the closest core.
    Drop any cores which have < 10% of the median number of cells.
    """
    # Group by the closest core and calculate the mean position
    # but only use the cells which are <= 2X the median distance
    median_dist = minor_grid_cells['closest_dist'].median()

    n_cells = (
        minor_grid_cells
        .query(f"closest_dist <= {2 * median_dist}")
        .groupby('closest_core')
        .size()
    )

    keep_cores = n_cells[n_cells >= 0.1 * n_cells.median()].index

    return (
        minor_grid_cells
        .loc[
            minor_grid_cells['closest_core'].isin(
                keep_cores
            )
        ]
        .query(f"closest_dist <= {2 * median_dist}")
        .groupby('closest_core')
        .agg({'grid_x': 'mean', 'grid_y': 'mean'})
    )


def _draw_hull(core_coords: pd.DataFrame):
    """
    Draw a convex hull around the points assigned to a core using scipy.spatial.ConvexHull
    """

    if core_coords.shape[0] < 3:
        return None  # Not enough points to form a hull

    # Calculate the distance of every point to the centroid
    mean_x = core_coords['x'].mean()
    mean_y = core_coords['y'].mean()
    core_coords = core_coords.assign(
        distance=lambda df: np.sqrt(((df['x'] - mean_x) ** 2) + ((df['y'] - mean_y) ** 2))
    )

    # Use the 99th percentile of the distance to filter out outliers
    # and draw the hull around the remaining points
    # This helps to avoid noise and outliers affecting the hull shape
    # We use a 10% margin to ensure we include all points within the hull
    max_distance = core_coords['distance'].quantile(0.99) * 1.1

    points = (
        core_coords
        .query(f'distance <= {max_distance}')
        [['x', 'y']]
        .values
    )
    hull = ConvexHull(points)

    return points[hull.vertices]


def find_tma_cores(
    points: SpatialPoints,
    angle: float,
    nrows: int = None,
    ncols: int = None,
    subsample_n=100000,
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
    x_grid = _find_grid(coords["x"].sample(subsample_n), ncols)
    y_grid = _find_grid(coords["y"].sample(subsample_n), nrows)

    cores = _find_cores(coords, x_grid, y_grid, min_prop_cells=min_prop_cells)

    # Rotate the cores back
    _rotate_cores(cores, -angle)

    return pd.DataFrame(cores)


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
    assert row_start in ["Top", "Bottom"], "row_start must be 'Top' or 'Bottom'"
    assert col_start in ["Left", "Right"], "col_start must be 'Left'' or 'Right'"

    row_map = _make_index_map(
        cores['row_i'],
        row_start == "Bottom",
        not rows_are_letters
    )
    print(row_map)

    col_map = _make_index_map(
        cores['col_i'],
        col_start == "Left",
        rows_are_letters
    )
    print(col_map)

    return cores.assign(
        name=cores.apply(
            lambda r: (
                f"{row_map[r['row_i']]}{col_map[r['col_i']]}"
                if rows_are_letters
                else f"{col_map[r['col_i']]}{row_map[r['row_i']]}"
            ),
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
            if not are_letters
            else range(1, 1+vals.shape[0])
        )
    ))
