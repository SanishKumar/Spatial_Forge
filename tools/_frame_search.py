"""Where a scan's frame sits in a model, found without a starting guess.

Registration refines a placement that is already nearly right. It has to be
given one, and a dataset that publishes its trajectory and its model in
unrelated frames does not give one. Two trajectories through the same room
can be anchored a metre and a half apart, and a guess that places the first
leaves the second in a basin the refinement never leaves.

A room is mostly planes in three perpendicular directions: floor and
ceiling, and two pairs of walls. That is enough to place it.

  - Depth gives oriented points: a normal from the differences across and
    down the image, turned toward the camera.
  - The three directions most normals lie along are found in the depth and
    in the model.
  - One triple of axes can be laid on the other in 24 ways. Each is a
    rotation.
  - Under each rotation, a plane facing along an axis sits at one position
    on it. The scan's positions, kept separately for the two ways a plane
    can face, are slid along the model's until as much of the scan as
    possible sits where the model has a plane facing the same way. Three
    axes give a translation.
  - Each placement is scored by the share of the depth that then lies near
    the model and faces the way the model faces there.

The best is a starting guess and nothing more. It is good to a histogram
bin and to however well the axes were found, which is inside what the
registration needs and far outside what a measurement can use. The
registration runs from it as from any other guess and has to settle.

It can be wrong in two ways, and both are refused rather than returned.
If nothing places most of the depth on the model, no placement was found.
If a second placement does nearly as well as the first, the scan cannot
tell them apart: a bare rectangular room looks the same turned half way
round.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from tools._nearest import NearestPointIndex

from spatialforge.tsdf_stream_fusion import _decode_metric_depth

# A normal counts as lying along an axis within this angle of it.
AXIS_TOLERANCE_DEG = 8.0
# Width of one bin of plane positions along an axis.
PLANE_BIN_M = 0.02
# The model has a plane at a position if at least this share of its points
# facing that way sit in the bin there.
MODEL_PLANE_SHARE = 0.002
# A depth sample supports a placement if it lands this near a model point
# whose normal is this close to its own.
NEAR_LIMIT_M = 0.05
FACING_COSINE = 0.8
# The best placement must put at least this share of the depth on the
# model, and no other may come within this ratio of it.
MIN_AGREEMENT = 0.5
AMBIGUITY_RATIO = 0.9
# Pixels either side of a sample that its normal is taken across. On a
# plane, at any slope, inverse depth is linear across the image, so the
# sample's is the mean of the two either side. A sample is taken to
# straddle an edge when it is further from that mean than this, relative
# to itself.
NORMAL_BASELINE_PIXELS = 4
NORMAL_PLANARITY_TOLERANCE = 0.03
# An axis every scan of a room shows must hold at least this share of the
# normals, or the scene is not the kind this search places.
MIN_AXIS_SHARE = 0.03

_AXIS_SAMPLE = 20_000
_AXIS_CANDIDATES = 1_000
_VOTE_CHUNK = 250
_SCORE_SAMPLE = 30_000

_AXIS_COSINE = math.cos(math.radians(AXIS_TOLERANCE_DEG))
_AXIS_SINE = math.sin(math.radians(AXIS_TOLERANCE_DEG))


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    """One placement of the scan in the model, and how well it fits."""

    model_from_source: np.ndarray
    agreement: float


def oriented_depth(
    session,
    observations,
    camera,
    depth_scale_m: float,
    pixel_step: int,
    source_from_session: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled depth points and unit normals toward the camera.

    In the source frame. A sample is kept only where the four pixels its
    normal is taken across lie, with it, on what could be one plane.
    """

    base = NORMAL_BASELINE_PIXELS
    rows = np.arange(base, camera.height - base, pixel_step)
    columns = np.arange(base, camera.width - base, pixel_step)
    if not len(rows) or not len(columns):
        raise SystemExit("the image is too small to take normals from")
    row, column = np.meshgrid(rows, columns, indexing="ij")
    across = (np.arange(camera.width) - camera.cx) / camera.fx
    down = (np.arange(camera.height) - camera.cy) / camera.fy

    points = []
    normals = []
    for observation in observations:
        depth = _decode_metric_depth(
            session, observation, camera.width, camera.height, depth_scale_m
        )
        depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)
        position = np.stack(
            [across[None, :] * depth, down[:, None] * depth, depth], axis=-1
        )
        centre = position[row, column]
        right = position[row, column + base]
        left = position[row, column - base]
        below = position[row + base, column]
        above = position[row - base, column]
        with np.errstate(invalid="ignore", divide="ignore"):
            normal = np.cross(right - left, below - above)
            length = np.linalg.norm(normal, axis=-1)
            keep = np.isfinite(length) & (length > 0.0)
            inverse = 1.0 / centre[..., 2]
            for one, other in ((right, left), (below, above)):
                bend = 0.5 * (1.0 / one[..., 2] + 1.0 / other[..., 2])
                keep &= (
                    np.abs(bend - inverse)
                    < NORMAL_PLANARITY_TOLERANCE * inverse
                )
        centre = centre[keep]
        normal = normal[keep] / length[keep][:, None]
        away = np.einsum("ij,ij->i", normal, centre) > 0.0
        normal[away] *= -1.0
        pose = np.array(
            [float(v) for v in observation.pose.data["T_world_camera"]]
        ).reshape((4, 4))
        to_source = source_from_session @ pose
        points.append(centre @ to_source[:3, :3].T + to_source[:3, 3])
        normals.append(normal @ to_source[:3, :3].T)
    if not points or not sum(len(part) for part in points):
        raise SystemExit("no sampled frame has depth a normal can be taken on")
    return np.concatenate(points), np.concatenate(normals)


def dominant_axes(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Three perpendicular directions most normals lie along.

    Returned as the columns of a rotation, with the share of the normals
    within the tolerance of each. Every normal votes for each of a spread
    of candidate directions it lies along, either way round; the winner is
    then moved to the mean of its voters. The second axis is found the
    same way among directions perpendicular to the first, and the third is
    their cross product.
    """

    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) < 3:
        raise ValueError("normals must be an N x 3 array of at least three")
    sample = normals[:: max(1, math.ceil(len(normals) / _AXIS_SAMPLE))]
    candidates = sample[:: max(1, math.ceil(len(sample) / _AXIS_CANDIDATES))]

    def along(direction: np.ndarray) -> np.ndarray:
        return sample @ direction

    def winner(allowed: np.ndarray) -> np.ndarray:
        chosen = candidates[allowed]
        if not len(chosen):
            raise SystemExit(
                "the surfaces do not face in three perpendicular "
                "directions; this scene cannot be placed by its planes"
            )
        votes = np.concatenate(
            [
                (
                    np.abs(sample @ chosen[first:first + _VOTE_CHUNK].T)
                    > _AXIS_COSINE
                ).sum(axis=0)
                for first in range(0, len(chosen), _VOTE_CHUNK)
            ]
        )
        direction = chosen[int(np.argmax(votes))]
        for _ in range(3):
            dots = along(direction)
            near = np.abs(dots) > _AXIS_COSINE
            direction = (sample[near] * np.sign(dots[near])[:, None]).sum(
                axis=0
            )
            direction = direction / np.linalg.norm(direction)
        return direction

    first = winner(np.ones(len(candidates), dtype=bool))
    second = winner(np.abs(candidates @ first) < _AXIS_SINE)
    second = second - (second @ first) * first
    second = second / np.linalg.norm(second)
    axes = np.stack([first, second, np.cross(first, second)], axis=1)
    shares = np.array(
        [
            float(np.mean(np.abs(along(axes[:, k])) > _AXIS_COSINE))
            for k in range(3)
        ]
    )
    if shares.min() < MIN_AXIS_SHARE:
        raise SystemExit(
            "the surfaces do not face in three perpendicular directions "
            f"(the three found hold {shares[0]:.3f}, {shares[1]:.3f} and "
            f"{shares[2]:.3f} of the normals); this scene cannot be placed "
            "by its planes"
        )
    return axes, shares


def proper_signed_permutations() -> list[np.ndarray]:
    """The 24 rotations that carry a triple of axes onto itself."""

    found = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            if np.linalg.det(matrix) > 0.0:
                found.append(matrix)
    return found


def _plane_positions(
    points: np.ndarray, normals: np.ndarray, axes: np.ndarray
) -> list[list[np.ndarray]]:
    """Per axis and facing, where the points facing that way sit on it."""

    positions = []
    for k in range(3):
        along = normals @ axes[:, k]
        place = points @ axes[:, k]
        positions.append(
            [place[sign * along > _AXIS_COSINE] for sign in (1.0, -1.0)]
        )
    return positions


def _histogram(values: np.ndarray, low: float, bins: int) -> np.ndarray:
    index = np.floor((values - low) / PLANE_BIN_M).astype(np.int64)
    return np.bincount(
        index[(index >= 0) & (index < bins)], minlength=bins
    )[:bins].astype(np.float64)


def _planes_present(values: np.ndarray, low: float, bins: int) -> np.ndarray:
    """Where the model has a plane: one in its bin, a half either side.

    A plane is a plane however much of it there is. A model of a room
    holds the outsides of its walls as well, which no scan from inside
    sees and which are larger than any wall with furniture against it; if
    size counted, the scan's walls would be laid on those.
    """

    counted = _histogram(values, low, bins)
    present = (counted >= max(1.0, MODEL_PLANE_SHARE * len(values))).astype(
        np.float64
    )
    beside = np.zeros(bins)
    beside[1:] = present[:-1]
    beside[:-1] = np.maximum(beside[:-1], present[1:])
    return np.maximum(present, 0.5 * beside)


def _best_shift(model: list[np.ndarray], source: list[np.ndarray]) -> float:
    """How far to slide the source's planes to lie on the model's.

    The slide that puts the most source points where the model has a
    plane facing the same way.
    """

    model_all = np.concatenate(model)
    source_all = np.concatenate(source)
    if not len(model_all) or not len(source_all):
        return 0.0
    model_low = float(model_all.min())
    source_low = float(source_all.min())
    model_bins = int((float(model_all.max()) - model_low) / PLANE_BIN_M) + 1
    source_bins = int((float(source_all.max()) - source_low) / PLANE_BIN_M) + 1
    explained = np.zeros(model_bins + source_bins - 1)
    for facing in range(2):
        explained += np.correlate(
            _planes_present(model[facing], model_low, model_bins),
            _histogram(source[facing], source_low, source_bins),
            mode="full",
        )
    # Entry j lays source bin n on model bin n + j - (source_bins - 1).
    lag = int(np.argmax(explained)) - (source_bins - 1)
    return model_low - source_low + lag * PLANE_BIN_M


def search_frame(
    points: np.ndarray,
    normals: np.ndarray,
    model_points: np.ndarray,
    model_normals: np.ndarray,
    index: NearestPointIndex,
) -> list[FrameCandidate]:
    """Every way the scan's planes lie on the model's, best first.

    ``index`` must be built on ``model_points``. Twenty-four candidates are
    returned; ``choose_frame`` says whether the first can be trusted.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    model_points = np.asarray(model_points, dtype=np.float64)
    model_normals = np.asarray(model_normals, dtype=np.float64)
    model_axes, _ = dominant_axes(model_normals)
    source_axes, _ = dominant_axes(normals)
    model_planes = _plane_positions(model_points, model_normals, model_axes)

    thin = max(1, math.ceil(len(points) / _SCORE_SAMPLE))
    candidates = []
    for permutation in proper_signed_permutations():
        rotation = model_axes @ permutation @ source_axes.T
        turned_points = points @ rotation.T
        turned_normals = normals @ rotation.T
        planes = _plane_positions(turned_points, turned_normals, model_axes)
        shift = np.array(
            [_best_shift(model_planes[k], planes[k]) for k in range(3)]
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = model_axes @ shift
        placed = turned_points[::thin] + transform[:3, 3]
        found, _ = index.nearest(placed, limit_m=NEAR_LIMIT_M)
        near = found >= 0
        facing = np.einsum(
            "ij,ij->i", turned_normals[::thin][near], model_normals[found[near]]
        )
        candidates.append(
            FrameCandidate(
                model_from_source=transform,
                agreement=float(np.count_nonzero(facing > FACING_COSINE))
                / len(placed),
            )
        )
    # Stable: equal scores keep the order the rotations were tried in.
    return sorted(candidates, key=lambda candidate: -candidate.agreement)


def choose_frame(candidates: list[FrameCandidate]) -> FrameCandidate:
    """The best placement, if it is one and the scan could tell.

    Refuses a best placement that leaves most of the depth off the model,
    and one that a second placement nearly matches.
    """

    best, second = candidates[0], candidates[1]
    if best.agreement < MIN_AGREEMENT:
        raise SystemExit(
            "no placement of the scan in the model was found: the best "
            f"puts {100 * best.agreement:.1f}% of the depth on it. Give an "
            "--initial-translation"
        )
    if second.agreement > AMBIGUITY_RATIO * best.agreement:
        raise SystemExit(
            "the scan fits the model about as well in two placements "
            f"({100 * best.agreement:.1f}% and "
            f"{100 * second.agreement:.1f}% of the depth); it cannot tell "
            "them apart. Give an --initial-translation"
        )
    return best
