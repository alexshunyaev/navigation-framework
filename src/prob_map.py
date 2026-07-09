import logging

import numpy as np

logger = logging.getLogger(__name__)


def _gaussian_blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """
    Separable 2-D Gaussian convolution, implemented from scratch.

    A 2-D isotropic Gaussian is separable into two 1-D convolutions (one per
    axis), which is what we exploit here rather than calling a library filter.
    The 1-D kernel is truncated at three standard deviations and re-normalised
    so the weights sum to one (preserving the field's overall magnitude), and
    the borders use reflect padding so obstacles near an edge are not diluted.
    """
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()

    def convolve_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        """Convolve ``arr`` with the 1-D Gaussian kernel along a single axis."""
        # 'symmetric' padding mirrors the edge sample (a b c | c b a), matching
        # the boundary handling of a standard Gaussian image filter.
        padded = np.pad(arr, [(radius, radius) if a == axis else (0, 0)
                              for a in range(arr.ndim)], mode="symmetric")
        out = np.zeros_like(arr)
        for offset, weight in enumerate(kernel):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(offset, offset + arr.shape[axis])
            out += weight * padded[tuple(sl)]
        return out

    return convolve_axis(convolve_axis(field, 0), 1)


class ProbabilisticMap:
    """
    Probabilistic occupancy map that models sensor uncertainty via a Gaussian risk field.

    Instead of a binary obstacle grid, each cell holds a risk value in [0, 1]:
    1 = certain obstacle, 0 = certainly free. The Gaussian kernel spreads
    uncertainty outward from known obstacle cells so the planner naturally
    prefers trajectories that pass far from any obstacle.
    """

    def __init__(self, grid: np.ndarray, sigma: float = 0.8,
                 collision_threshold: float = 0.5):
        """
        Args:
            grid: Binary occupancy grid (0=free, 1=obstacle).
            sigma: Gaussian spread in grid cells — larger values encode more uncertainty.
            collision_threshold: Risk level above which a cell is treated as a collision.
        """
        self.grid = grid
        self.sigma = sigma
        self.collision_threshold = collision_threshold
        self.risk_field = self._build_risk_field()

    def _build_risk_field(self) -> np.ndarray:
        """Blur the binary grid into a [0, 1] risk field, pinning known obstacles to 1."""
        blurred = _gaussian_blur(self.grid.astype(float), self.sigma)
        # Pin known obstacle cells to 1.0; Gaussian gradient handles the uncertainty margin
        return np.clip(np.maximum(blurred, self.grid), 0.0, 1.0)

    def planning_grid(self) -> np.ndarray:
        """
        Binary grid for the *global* planner that agrees with the local planner.

        A* originally ran on the raw occupancy grid and would happily route the
        path through single free cells wedged between obstacles. The local
        planner, however, collision-checks against this inflated risk field and
        treats any cell at/above ``collision_threshold`` as lethal — so those
        corridors are impassable and the robot gets stuck. Planning A* on this
        grid instead guarantees the global path only visits cells the local
        planner also considers safe.
        """
        return (self.risk_field >= self.collision_threshold).astype(int)

    def get_risk(self, r: float, c: float) -> float:
        """Query risk at continuous coordinates via bilinear interpolation."""
        return float(self.get_risk_batch(np.array([[r, c]]))[0])

    def get_risk_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Vectorised bilinear-interpolated risk lookup for an array of (r, c) positions.

        The risk field is a smooth Gaussian-blurred field; snapping continuous
        coordinates to the nearest cell introduces artificial discontinuities at
        cell boundaries. Those discontinuities can dwarf the DWA velocity-cost
        incentive to move, causing the optimiser to get stuck preferring near-zero
        velocity ("frozen robot"). Bilinear interpolation keeps the cost continuous
        in (v, omega), so it varies smoothly with the candidate trajectory.
        """
        rows, cols = self.risk_field.shape
        r = np.clip(positions[:, 0], 0, rows - 1)
        c = np.clip(positions[:, 1], 0, cols - 1)

        r0 = np.floor(r).astype(int)
        c0 = np.floor(c).astype(int)
        r1 = np.minimum(r0 + 1, rows - 1)
        c1 = np.minimum(c0 + 1, cols - 1)

        fr = r - r0
        fc = c - c0

        v00 = self.risk_field[r0, c0]
        v01 = self.risk_field[r0, c1]
        v10 = self.risk_field[r1, c0]
        v11 = self.risk_field[r1, c1]

        return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc
                + v10 * fr * (1 - fc) + v11 * fr * fc)

    def calc_trajectory_cost(self, trajectory: np.ndarray) -> float:
        """
        Probabilistic obstacle cost for a candidate trajectory.

        Returns inf if the maximum risk along the trajectory exceeds the
        collision threshold; otherwise returns a penalty that rises
        quadratically toward the threshold, so the planner is discouraged
        from hugging obstacles well before that becomes outright infeasible,
        without ever outweighing every other cost term and freezing the
        robot in front of a legitimately narrow (but safe) passage.
        """
        max_risk = float(np.max(self.get_risk_batch(trajectory[:, :2])))
        if max_risk >= self.collision_threshold:
            return float("inf")
        return (max_risk / self.collision_threshold) ** 2
