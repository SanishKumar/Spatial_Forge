"""A route's certificate, held to the free-space map and to block proofs.

Three things are claimed of a certificate, and each is held to a path that
already exists.

Which voxels a route means, and whether they are clear, is the free-space
map's rule. So what a volume says of a route is worked out a second time
here from the map tool's own dense slab and its own column verdicts, for
routes drawn at random on all three axes, and the two must agree to the
voxel.

That the blocks follow from the scan is the block proof's business. So the
blocks and tiles of a certificate are compared with the single proofs of
the same blocks, and with the blocks fused alone from whole frames.

And a certificate must not be made to say what is not so. A map with an
obstacle taken out of the way says its route is clear; the certificate of
that route says refuted. A certificate altered under an honest statement
is invalid, or else it is a certificate of some other route and says
truly what the map holds for that one.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import struct
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from spatialforge import (
    MAX_TSDF_ROUTE_BLOCKS,
    MAX_TSDF_ROUTE_CHAIN_COLUMNS,
    MAX_TSDF_ROUTE_RADIUS,
    MAX_TSDF_ROUTE_WAYPOINTS,
    TSDF_ROUTE_CHECKS,
    TSDF_ROUTE_MIN_WEIGHT,
    TsdfProofFrameTiles,
    TsdfRoute,
    TsdfRouteCertificate,
    TsdfRouteClearance,
    TsdfRouteVerdict,
    certify_tsdf_route,
    check_tsdf_block_proof,
    check_tsdf_route_certificate,
    commit_tsdf_scan,
    commit_tsdf_volume,
    decode_tsdf_route_certificate,
    fuse_tsdf_blocks_alone,
    load_tsdf_block_plan,
    load_tsdf_block_volume,
    load_tsdf_route_certificate,
    measure_tsdf_route,
    prove_tsdf_blocks,
    state_tsdf_map,
    tsdf_volume_merkle_tree,
    write_tsdf_block_volume,
    write_tsdf_map_statement,
    write_tsdf_route_certificate,
)
from spatialforge import tsdf_route_certificate
from spatialforge.cli import main
from spatialforge.errors import TsdfError
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_evidence import (
    _fuse_blocks_from_frames,
    _gather_pooled_frames,
)
from spatialforge.tsdf_commitment import _statement
from spatialforge.tsdf_expanded_plan import write_tsdf_expanded_block_plan
from spatialforge.tsdf_stream_expansion import (
    propose_tsdf_plan_expansion_streaming,
)

from tests.heavy_fixtures import shared_room_case
from tests.test_tsdf_block_evidence import forbidden_calls, fused
from tests.test_tsdf_block_evidence import tree_snapshot
from tests.test_tsdf_block_proof import (
    change_one_sample,
    drop_a_frame,
    fixture_volume,
    forge,
    frozen,
    rows_that_differ,
    tile_of_the_scan,
    with_frame,
)
from tests.test_tsdf_block_volume import assemble, split
from tests.test_tsdf_commitment import copied
from tools.free_space_map import FREE, band_slab, classify_columns

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"
FIXTURE_STATEMENT_SHA256 = (
    "7b771f1ef5ff5249f48cf29affdbce16df2349f02b67cf7d9accf4884414e780"
)
CHECKS = (
    "statement",
    "shape",
    "route",
    "scan root",
    "map root",
    "depth roots",
    "poses",
    "pixels",
    "fusion",
    "clearance",
)

# Through the committed fixture, whose two frames have seen a voxel twice
# at most: a route asks for two observations there. Three columns in
# front of the wall, and the same line carried on into it.
FIXTURE_CLEAR = TsdfRoute(2, 0, 0, 0, 2, ((2, 0), (5, 0)))
FIXTURE_CLEAR_SHA256 = (
    "a755eb4b31d49fdc3a2011d288044fd4657d39be2a998a0a938074fbc59cf30f"
)
FIXTURE_CLEAR_BYTES = 7189
FIXTURE_WALL = TsdfRoute(2, 0, 0, 0, 2, ((5, 0), (10, 0)))
FIXTURE_WALL_SHA256 = (
    "7bfed8422423d5d0211b7b9042099429bc545f8263a9d566bdfe4a7703a9ecb7"
)
FIXTURE_FOUR = TsdfRoute(2, -1, 0, 0, 2, ((3, -1), (6, 0)))
FIXTURE_FOUR_SHA256 = (
    "4b6bf1750f70bacd00c3bc794509fef832ca94ca3c991c5d0de5c24dc444d81a"
)
FIXTURE_CLEAR_HEADER = "\n".join(
    (
        "{",
        '  "blocks": {',
        '    "held": 1,',
        '    "of": 8',
        "  },",
        '  "frames": 2,',
        '  "image": {',
        '    "height": 2,',
        '    "tile_pixels": 16,',
        '    "width": 2',
        "  },",
        '  "route": {',
        '    "band_first": 0,',
        '    "band_last": 0,',
        '    "min_weight": 2,',
        '    "radius": 0,',
        '    "up_axis": "z",',
        '    "waypoints": 2',
        "  },",
        '  "schema": "spatialforge.tsdf-route-certificate",',
        '  "schema_version": "0.1.0",',
        f'  "statement_sha256": "{FIXTURE_STATEMENT_SHA256}",',
        '  "tile_frames": 2',
        "}",
        "",
    )
).encode("ascii")

# Through the generated room, expanded into the space it saw empty. At
# the camera's height, across the open middle; and lower down, from open
# floor into the side of the box.
ROOM_CLEAR = TsdfRoute(2, -5, 5, 2, 3, ((20, -10), (50, -10), (50, 20)))
ROOM_BLOCKED = TsdfRoute(2, -20, -10, 2, 3, ((40, -28), (40, -11)))
ROOM_UNKNOWN = TsdfRoute(2, -5, 5, 2, 3, ((0, 0), (5, 0)))

_FORBIDDEN_TO_A_CHECK = (
    "spatialforge.tsdf_route_certificate._gather_pooled_frames",
    "spatialforge.tsdf_route_certificate._commit_tsdf_scan",
    "spatialforge.tsdf_route_certificate.tsdf_volume_merkle_tree",
    "spatialforge.tsdf_block_evidence.replay_session",
    "spatialforge.tsdf_commitment.replay_session",
    "spatialforge.tsdf_stream_fusion._read_depth_array",
    "spatialforge.tsdf_block_evidence._decode_raw_depth",
    "spatialforge.tsdf_commitment._decode_raw_depth",
    "pathlib.Path.open",
    "PIL.Image.open",
)

WORK: SimpleNamespace | None = None
_GONE = object()


def swept_by_hand(route: TsdfRoute) -> set[tuple[int, int]]:
    """Every column within the radius of the route's line, one at a time."""

    reach = range(-route.radius, route.radius + 1)
    return {
        (a + da, b + db)
        for a, b in route.chain().tolist()
        for da in reach
        for db in reach
        if da * da + db * db <= route.radius * route.radius
    }


def blocks_by_hand(route: TsdfRoute) -> list[tuple[int, int, int]]:
    """Every block holding a voxel of the route, one voxel at a time."""

    first, second = route.across_axes
    blocks = set()
    for a, b in swept_by_hand(route):
        for height in range(route.band_first, route.band_last + 1):
            voxel = [0, 0, 0]
            voxel[first], voxel[second], voxel[route.up_axis] = a, b, height
            blocks.add(tuple(value // 8 for value in voxel))
    return sorted(blocks, key=lambda block: block[::-1])


def what_the_map_tool_says(volume, route: TsdfRoute):
    """A route's clearance from the free-space map's own slab and verdicts.

    Returns the count of unobserved voxels, of occupied ones, the first
    voxel that is not clear, and whether the tool calls every column of
    the route free.
    """

    voxel = volume.voxel_size_m
    values, weights, origin, across = band_slab(
        volume,
        route.up_axis,
        route.band_first * voxel,
        (route.band_last + 1) * voxel,
    )
    if (
        tuple(across) != route.across_axes
        or origin[2] != route.band_first
        or values.shape[2] != route.band_voxels
    ):
        raise AssertionError("the slab is not the route's band")
    verdicts = classify_columns(values, weights, min_weight=route.min_weight)
    unobserved = occupied = 0
    every_column_free = True
    not_clear = []
    for a, b in sorted(swept_by_hand(route)):
        i, j = a - origin[0], b - origin[1]
        if 0 <= i < values.shape[0] and 0 <= j < values.shape[1]:
            known = weights[i, j] >= route.min_weight
            behind = known & (values[i, j] <= 0.0)
            unobserved += int(np.count_nonzero(~known))
            occupied += int(np.count_nonzero(behind))
            heights = np.flatnonzero(~known | behind).tolist()
            every_column_free &= bool(verdicts[i, j] == FREE)
        else:
            unobserved += route.band_voxels
            heights = list(range(route.band_voxels))
            every_column_free = False
        for height in heights:
            at = [0, 0, 0]
            at[across[0]], at[across[1]] = a, b
            at[route.up_axis] = route.band_first + height
            not_clear.append(tuple(at))
    first = min(not_clear, key=lambda at: at[::-1]) if not_clear else None
    return unobserved, occupied, first, every_column_free


def say_the_way_is_clear(route: TsdfRoute):
    """An edit of a map: every voxel of the route seen, and empty."""

    swept = swept_by_hand(route)

    def edit(blocks, sums, weights) -> None:
        for row, (bx, by, bz) in enumerate(blocks.tolist()):
            for z in range(8):
                if not route.band_first <= bz * 8 + z <= route.band_last:
                    continue
                for y in range(8):
                    for x in range(8):
                        if (bx * 8 + x, by * 8 + y) in swept:
                            weight = max(
                                int(weights[row, z, y, x]), route.min_weight
                            )
                            weights[row, z, y, x] = weight
                            sums[row, z, y, x] = float(weight)

    return edit


def without_block(certificate, place: int):
    """A certificate with one of its blocks taken out."""

    keep = [
        index for index in range(certificate.held_block_count)
        if index != place
    ]
    return replace(
        certificate,
        block_rows=tuple(certificate.block_rows[index] for index in keep),
        block_indices=tuple(
            certificate.block_indices[index] for index in keep
        ),
        tsdf_sums=frozen(certificate.tsdf_sums[keep]),
        weights=frozen(certificate.weights[keep]),
        block_paths=tuple(certificate.block_paths[index] for index in keep),
    )


def with_tile_frame(certificate, position: int, **changes):
    return replace(
        certificate,
        tile_frames=(
            certificate.tile_frames[:position]
            + (replace(certificate.tile_frames[position], **changes),)
            + certificate.tile_frames[position + 1:]
        ),
    )


def rebuilt(encoded: bytes, change) -> bytes:
    """A certificate file with its header document changed."""

    (length,) = struct.unpack_from("<Q", encoded, 8)
    document = json.loads(encoded[16:16 + length])
    change(document)
    header = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    return (
        encoded[:8] + struct.pack("<Q", len(header)) + header
        + encoded[16 + length:]
    )


def setUpModule() -> None:
    global WORK
    root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
    unittest.addModuleCleanup(shutil.rmtree, root, True)

    volume_path, volume, session = fixture_volume(root, "fixture", FIXTURE)
    statement = state_tsdf_map(volume, session)
    statement_path = root / "fixture.sftcommit"
    write_tsdf_map_statement(statement, statement_path)
    header, indices, sums, weights = split(volume_path.read_bytes())
    target = int(np.flatnonzero(weights)[0])
    sums[target] = -sums[target]
    forged_path = root / "forged.sftvol"
    forged_path.write_bytes(assemble(header, indices, sums, weights))

    room = shared_room_case()
    expanded_path = root / "room-expanded.sftplan"
    write_tsdf_expanded_block_plan(
        room.plan,
        propose_tsdf_plan_expansion_streaming(room.plan, room.session),
        expanded_path,
    )
    room_storage, room_receipt = fused(
        load_tsdf_block_plan(expanded_path), room.session
    )
    room_volume_path = root / "room.sftvol"
    write_tsdf_block_volume(room_storage, room_receipt, room_volume_path)
    room_volume = load_tsdf_block_volume(room_volume_path)
    room_statement = state_tsdf_map(room_volume, room.session)

    WORK = SimpleNamespace(
        root=root,
        session=session,
        volume_path=volume_path,
        volume=volume,
        statement=statement,
        statement_path=statement_path,
        forged_path=forged_path,
        fixture_clear=certify_tsdf_route(
            statement, volume, session, FIXTURE_CLEAR
        ),
        room=room,
        room_volume_path=room_volume_path,
        room_volume=room_volume,
        room_statement=room_statement,
        room_clear=certify_tsdf_route(
            room_statement, room_volume, room.session, ROOM_CLEAR
        ),
        room_blocked=certify_tsdf_route(
            room_statement, room_volume, room.session, ROOM_BLOCKED
        ),
    )


class RouteGeometryTests(unittest.TestCase):
    def test_a_line_between_waypoints_written_out(self) -> None:
        def chain(*waypoints):
            return [
                tuple(column)
                for column in TsdfRoute(2, 0, 0, 0, 3, waypoints)
                .chain()
                .tolist()
            ]

        self.assertEqual(chain((4, -7)), [(4, -7)])
        self.assertEqual(
            chain((0, 0), (3, 1)), [(0, 0), (1, 0), (2, 1), (3, 1)]
        )
        self.assertEqual(
            chain((3, 1), (0, 0)), [(3, 1), (2, 1), (1, 0), (0, 0)]
        )
        self.assertEqual(
            chain((0, 0), (-2, -2), (-2, 1)),
            [(0, 0), (-1, -1), (-2, -2), (-2, -1), (-2, 0), (-2, 1)],
        )
        self.assertEqual(
            chain((2, 0), (5, 0)), [(2, 0), (3, 0), (4, 0), (5, 0)]
        )

    def test_a_line_keeps_to_the_straight_line_and_never_jumps(self) -> None:
        draw = random.Random(20261012)
        for _ in range(300):
            start = (draw.randint(-60, 60), draw.randint(-60, 60))
            end = (draw.randint(-60, 60), draw.randint(-60, 60))
            if start == end:
                continue
            chain = TsdfRoute(2, 0, 0, 0, 3, (start, end)).chain().tolist()
            run = (end[0] - start[0], end[1] - start[1])
            long_axis = 0 if abs(run[0]) >= abs(run[1]) else 1
            short_axis = 1 - long_axis
            with self.subTest(start=start, end=end):
                self.assertEqual(tuple(chain[0]), start)
                self.assertEqual(tuple(chain[-1]), end)
                # One column for each step along the longer side.
                self.assertEqual(len(chain), abs(run[long_axis]) + 1)
                for before, after in zip(chain, chain[1:]):
                    self.assertEqual(
                        abs(after[long_axis] - before[long_axis]), 1
                    )
                    self.assertLessEqual(
                        abs(after[short_axis] - before[short_axis]), 1
                    )
                # And never more than half a voxel off the true line
                # across it, in whole numbers:
                # |2 (s - s0) L - 2 (l - l0) S| <= |L|.
                for column in chain:
                    off = 2 * (
                        (column[short_axis] - start[short_axis])
                        * run[long_axis]
                        - (column[long_axis] - start[long_axis])
                        * run[short_axis]
                    )
                    self.assertLessEqual(abs(off), abs(run[long_axis]))

    def test_columns_and_blocks_are_what_one_voxel_at_a_time_gives(
        self,
    ) -> None:
        draw = random.Random(5)
        routes = [ROOM_CLEAR, ROOM_BLOCKED, ROOM_UNKNOWN, FIXTURE_FOUR]
        for up_axis in (0, 1, 2):
            for _ in range(8):
                first = draw.randint(-40, 30)
                routes.append(
                    TsdfRoute(
                        up_axis,
                        first,
                        first + draw.randint(0, 20),
                        draw.randint(0, 5),
                        3,
                        tuple(
                            (
                                draw.randint(-70, 70) + 200 * step,
                                draw.randint(-70, 70),
                            )
                            for step in range(draw.randint(1, 4))
                        ),
                    )
                )
        for route in routes:
            with self.subTest(route=route):
                swept = swept_by_hand(route)
                columns = route.columns()
                self.assertEqual(
                    [tuple(column) for column in columns.tolist()],
                    sorted(swept),
                )
                self.assertEqual(
                    list(route.block_indices()), blocks_by_hand(route)
                )
                self.assertEqual(
                    route.across_axes,
                    tuple(
                        axis for axis in (0, 1, 2) if axis != route.up_axis
                    ),
                )
                self.assertEqual(
                    route.band_voxels, route.band_last - route.band_first + 1
                )
        # A radius of nothing is the line itself, and a radius is a
        # disc: at two voxels, thirteen columns.
        disc = TsdfRoute(2, 0, 0, 2, 3, ((0, 0),))
        self.assertEqual(len(disc.columns()), 13)
        self.assertEqual(len(ROOM_CLEAR.chain()), 61)
        self.assertEqual(len(ROOM_CLEAR.columns()), 310)
        self.assertEqual(len(ROOM_CLEAR.block_indices()), 26)

    def test_routes_validate_themselves(self) -> None:
        good = ROOM_CLEAR
        cases = {
            "an up axis there is not": ({"up_axis": 3}, "up axis"),
            "an up axis by name": ({"up_axis": "z"}, "up axis"),
            "an up axis that is a flag": ({"up_axis": True}, "up axis"),
            "a band upside down": (
                {"band_first": 5, "band_last": -5},
                "at least one voxel",
            ),
            "a band in metres": ({"band_first": -0.2}, "voxel indices"),
            "a band out of reach": ({"band_last": 2**30 + 1}, "voxel indices"),
            "a radius below nothing": ({"radius": -1}, "radius"),
            "a radius too wide": (
                {"radius": MAX_TSDF_ROUTE_RADIUS + 1},
                "radius is a count of voxels from 0 to 64",
            ),
            "a radius in metres": ({"radius": 0.25}, "radius"),
            "no observations asked": ({"min_weight": 0}, "minimum weight"),
            "observations as a flag": ({"min_weight": True}, "minimum weight"),
            "no waypoints": ({"waypoints": ()}, "from 1 to 4096 waypoints"),
            "waypoints as a list": (
                {"waypoints": [(0, 0), (1, 1)]},
                "from 1 to 4096 waypoints",
            ),
            "too many waypoints": (
                {
                    "waypoints": tuple(
                        (index % 2, 0)
                        for index in range(MAX_TSDF_ROUTE_WAYPOINTS + 1)
                    )
                },
                "from 1 to 4096 waypoints",
            ),
            "a waypoint of three": (
                {"waypoints": ((0, 0, 0),)},
                "waypoint is two voxel indices",
            ),
            "a waypoint as a list": (
                {"waypoints": ([0, 0],)},
                "waypoint is two voxel indices",
            ),
            "a waypoint in metres": (
                {"waypoints": ((0.5, 0),)},
                "waypoint is two voxel indices",
            ),
            "a waypoint out of reach": (
                {"waypoints": ((0, -(2**30) - 1),)},
                "waypoint is two voxel indices",
            ),
            "a waypoint twice running": (
                {"waypoints": ((0, 0), (4, 4), (4, 4))},
                "must differ from the one before it",
            ),
            "a line too long": (
                {"waypoints": ((0, 0), (MAX_TSDF_ROUTE_CHAIN_COLUMNS, 3))},
                "runs through at most 65536 columns",
            ),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(good, **changes)
        # The longest line there may be is drawn; a wide one that long
        # is refused when its columns are asked for, and one that
        # touches too many blocks when its blocks are.
        longest = TsdfRoute(
            2, 0, 0, 0, 3, ((0, 0), (MAX_TSDF_ROUTE_CHAIN_COLUMNS - 1, 9))
        )
        self.assertEqual(len(longest.chain()), MAX_TSDF_ROUTE_CHAIN_COLUMNS)
        with self.assertRaisesRegex(TsdfError, "sweeps more columns"):
            replace(longest, radius=20).columns()
        self.assertEqual(MAX_TSDF_ROUTE_BLOCKS, 65_536)
        with self.assertRaisesRegex(
            TsdfError, "a certificate holds at most 65536"
        ):
            replace(longest, band_first=-600, band_last=600).block_indices()
        self.assertEqual(TSDF_ROUTE_MIN_WEIGHT, 3)


class ClearanceTests(unittest.TestCase):
    def test_a_volume_says_of_a_route_what_the_map_tool_says(self) -> None:
        volume = WORK.room_volume
        draw = random.Random(11)
        routes = [ROOM_CLEAR, ROOM_BLOCKED, ROOM_UNKNOWN]
        # The volume spans voxels -8..71, -40..39 and -32..39; these go
        # a little outside it on every side.
        spans = ((-12, 75), (-44, 43), (-36, 43))
        for up_axis in (0, 1, 2):
            first_axis, second_axis = (
                axis for axis in (0, 1, 2) if axis != up_axis
            )
            for _ in range(10):
                low = draw.randint(*spans[up_axis])
                routes.append(
                    TsdfRoute(
                        up_axis,
                        low,
                        min(low + draw.randint(0, 14), spans[up_axis][1]),
                        draw.randint(0, 3),
                        draw.choice((1, 3, 3, 6)),
                        tuple(
                            dict.fromkeys(
                                (
                                    draw.randint(*spans[first_axis]),
                                    draw.randint(*spans[second_axis]),
                                )
                                for _ in range(draw.randint(1, 4))
                            )
                        ),
                    )
                )
        # And some short ones in the open middle of the room, at the
        # camera's height, where there is something clear to find.
        for _ in range(8):
            low = draw.randint(-5, 2)
            routes.append(
                TsdfRoute(
                    2,
                    low,
                    low + draw.randint(0, 3),
                    draw.randint(0, 2),
                    3,
                    tuple(
                        dict.fromkeys(
                            (draw.randint(28, 56), draw.randint(-20, 20))
                            for _ in range(draw.randint(1, 3))
                        )
                    ),
                )
            )
        clear = 0
        for route in routes:
            with self.subTest(route=route):
                clearance, absent = measure_tsdf_route(volume, route)
                unobserved, occupied, first, free = what_the_map_tool_says(
                    volume, route
                )
                self.assertEqual(
                    clearance,
                    TsdfRouteClearance(
                        column_count=len(swept_by_hand(route)),
                        voxel_count=len(swept_by_hand(route))
                        * route.band_voxels,
                        unobserved_voxels=unobserved,
                        occupied_voxels=occupied,
                        first_voxel_not_clear=first,
                    ),
                )
                self.assertEqual(clearance.is_clear, free)
                self.assertEqual(
                    list(absent),
                    [
                        block
                        for block in blocks_by_hand(route)
                        if block not in volume.block_indices
                    ],
                )
                clear += clearance.is_clear
        # Of the forty-one, some are clear and most are not.
        self.assertEqual(len(routes), 41)
        self.assertGreater(clear, 5)
        self.assertLess(clear, 20)

    def test_the_named_routes_written_out(self) -> None:
        volume = WORK.room_volume
        self.assertEqual(volume.block_count, 524)
        self.assertEqual(
            measure_tsdf_route(volume, ROOM_CLEAR),
            (TsdfRouteClearance(310, 3410, 0, 0, None), ()),
        )
        self.assertEqual(
            measure_tsdf_route(volume, ROOM_BLOCKED),
            (TsdfRouteClearance(98, 1078, 222, 14, (39, -29, -20)), ()),
        )
        clearance, absent = measure_tsdf_route(volume, ROOM_UNKNOWN)
        self.assertFalse(clearance.is_clear)
        self.assertEqual(len(ROOM_UNKNOWN.block_indices()), 8)
        self.assertEqual(len(absent), 3)
        self.assertEqual(absent[0], (-1, -1, -1))
        self.assertEqual(
            measure_tsdf_route(WORK.volume, FIXTURE_CLEAR),
            (TsdfRouteClearance(4, 4, 0, 0, None), ()),
        )
        self.assertEqual(
            measure_tsdf_route(WORK.volume, FIXTURE_WALL),
            (TsdfRouteClearance(6, 6, 0, 3, (8, 0, 0)), ()),
        )
        # Asked for three observations, where the fixture has two.
        self.assertEqual(
            measure_tsdf_route(
                WORK.volume, replace(FIXTURE_CLEAR, min_weight=3)
            )[0],
            TsdfRouteClearance(4, 4, 4, 0, (2, 0, 0)),
        )
        with self.assertRaisesRegex(TsdfError, "TsdfBlockVolume"):
            measure_tsdf_route(WORK.volume_path, FIXTURE_CLEAR)
        with self.assertRaisesRegex(TsdfError, "TsdfRoute"):
            measure_tsdf_route(WORK.volume, ((2, 0), (5, 0)))


class CertificateTests(unittest.TestCase):
    def test_a_clear_route_is_certified_and_checked(self) -> None:
        certificate = WORK.room_clear
        self.assertEqual(TSDF_ROUTE_CHECKS, CHECKS)
        self.assertEqual(certificate.route, ROOM_CLEAR)
        self.assertEqual(
            list(certificate.block_indices), blocks_by_hand(ROOM_CLEAR)
        )
        self.assertEqual(
            (
                certificate.held_block_count,
                certificate.block_count,
                certificate.frame_count,
                len(certificate.tile_frames),
                certificate.tile_count,
                certificate.depth_byte_count,
                certificate.tile_path_hash_count,
                certificate.block_path_hash_count,
            ),
            (26, 524, 20, 20, 204, 104_448, 741, 260),
        )
        encoded = certificate.encoded()
        self.assertEqual(len(encoded), 302_899)
        decoded = decode_tsdf_route_certificate(encoded)
        self.assertIsInstance(decoded, TsdfRouteCertificate)
        self.assertEqual(decoded.encoded(), encoded)
        self.assertEqual(
            certificate.digest_sha256, hashlib.sha256(encoded).hexdigest()
        )
        check = check_tsdf_route_certificate(WORK.room_statement, decoded)
        self.assertIs(check.verdict, TsdfRouteVerdict.CLEAR)
        self.assertTrue(check.holds)
        self.assertIsNone(check.failed_check)
        self.assertIsNone(check.detail)
        self.assertEqual(check.passed_checks, CHECKS)
        self.assertEqual(check.refuted_blocks, ())
        self.assertEqual(
            check.clearance, TsdfRouteClearance(310, 3410, 0, 0, None)
        )
        self.assertEqual(
            (
                check.block_count,
                check.selected_observations,
                check.ready_observations,
            ),
            (26, 20, 20),
        )
        self.assertEqual(
            check.statement_digest_sha256,
            WORK.room_statement.digest_sha256,
        )

    def test_a_route_that_is_not_clear_is_certified_as_not_clear(
        self,
    ) -> None:
        certificate = WORK.room_blocked
        self.assertEqual(certificate.held_block_count, 12)
        self.assertEqual(len(certificate.encoded()), 106_846)
        check = check_tsdf_route_certificate(WORK.room_statement, certificate)
        self.assertIs(check.verdict, TsdfRouteVerdict.NOT_CLEAR)
        self.assertFalse(check.holds)
        self.assertEqual(check.failed_check, "clearance")
        self.assertEqual(check.passed_checks, CHECKS[:-1])
        self.assertEqual(
            check.detail,
            "222 of 1078 voxels of the route were observed fewer than 3 "
            "times and 14 are at or behind a surface, the first of them "
            "voxel (39, -29, -20)",
        )
        self.assertEqual(check.refuted_blocks, ())
        self.assertEqual(
            check.clearance,
            TsdfRouteClearance(98, 1078, 222, 14, (39, -29, -20)),
        )

    def test_its_blocks_and_tiles_are_the_single_proofs_of_them(
        self,
    ) -> None:
        for name, certificate in (
            ("clear", WORK.room_clear),
            ("blocked", WORK.room_blocked),
        ):
            with self.subTest(route=name):
                proofs = prove_tsdf_blocks(
                    WORK.room_statement,
                    WORK.room_volume,
                    WORK.room.session,
                    certificate.block_indices,
                )
                pooled: dict[int, dict] = {}
                for place, proof in enumerate(proofs):
                    self.assertIs(
                        check_tsdf_block_proof(
                            WORK.room_statement, proof
                        ).holds,
                        True,
                    )
                    self.assertEqual(proof.frames, certificate.frames)
                    self.assertEqual(
                        proof.block_row, certificate.block_rows[place]
                    )
                    self.assertEqual(
                        proof.block_path, certificate.block_paths[place]
                    )
                    self.assertEqual(
                        proof.tsdf_sums.tobytes(),
                        certificate.tsdf_sums[place].tobytes(),
                    )
                    self.assertEqual(
                        proof.weights.tobytes(),
                        certificate.weights[place].tobytes(),
                    )
                    for tiles in proof.tile_frames:
                        of_frame = pooled.setdefault(tiles.sequence, {})
                        for tile, path in zip(tiles.tiles.tiles, tiles.paths):
                            of_frame[tile[:2]] = (tile[2].tobytes(), path)
                # Each tile once, however many of the blocks read it.
                self.assertEqual(
                    {
                        tiles.sequence: {
                            tile[:2]: (tile[2].tobytes(), path)
                            for tile, path in zip(
                                tiles.tiles.tiles, tiles.paths
                            )
                        }
                        for tiles in certificate.tile_frames
                    },
                    pooled,
                )
                self.assertLess(
                    certificate.tile_count,
                    sum(proof.tile_count for proof in proofs),
                )
                # And the frames' leaves once. Together that is less
                # than half of what the proofs come to one by one.
                self.assertLess(
                    len(certificate.encoded()),
                    sum(len(proof.encoded()) for proof in proofs) // 2,
                )
                # And they are the blocks whole frames give.
                lone = fuse_tsdf_blocks_alone(
                    WORK.room.session,
                    certificate.block_indices,
                    voxel_size_m=WORK.room_volume.voxel_size_m,
                    truncation_m=WORK.room_volume.truncation_m,
                    frame_stride=WORK.room_volume.frame_stride,
                )
                for place, block in enumerate(lone):
                    self.assertEqual(
                        block.tsdf_sums.tobytes(),
                        certificate.tsdf_sums[place].tobytes(),
                    )
                    self.assertEqual(
                        block.weights.tobytes(),
                        certificate.weights[place].tobytes(),
                    )

    def test_every_block_fused_from_shared_tiles_is_the_block_fused_alone(
        self,
    ) -> None:
        # What a certificate's check rests on, asked of the whole room
        # at once: every frame's tiles gathered once for all 524 blocks,
        # and all of them fused from that.
        volume = WORK.room_volume
        options = {
            "voxel_size_m": volume.voxel_size_m,
            "truncation_m": volume.truncation_m,
            "frame_stride": volume.frame_stride,
        }
        frames = _gather_pooled_frames(
            WORK.room.session, volume.block_indices, tile_pixels=16, **options
        )
        self.assertEqual([frame.sequence for frame in frames], list(range(20)))
        before = tree_snapshot(WORK.room.session_path)
        with forbidden_calls(_FORBIDDEN_TO_A_CHECK):
            fused_together = _fuse_blocks_from_frames(
                WORK.room_statement.camera,
                WORK.room_statement.depth_scale_m,
                volume.voxel_size_m,
                volume.truncation_m,
                volume.block_indices,
                frames,
            )
        self.assertEqual(tree_snapshot(WORK.room.session_path), before)
        self.assertEqual(
            fused_together.sums.tobytes(), volume.tsdf_sums.tobytes()
        )
        self.assertEqual(
            fused_together.weights.tobytes(), volume.weights.tobytes()
        )
        lone = fuse_tsdf_blocks_alone(
            WORK.room.session, volume.block_indices, **options
        )
        self.assertEqual(fused_together.ready_observations, 20)
        for row, block in enumerate(lone):
            self.assertEqual(
                (
                    int(fused_together.ruled_out[row]),
                    int(fused_together.evaluated[row]),
                    int(fused_together.contributing[row]),
                ),
                (
                    block.ruled_out_observations,
                    block.evaluated_observations,
                    block.contributing_observations,
                ),
            )
        # A tile taken from any frame is missed, and named.
        for position, frame in enumerate(frames):
            if frame.tiles is None or frame.tiles.tile_count < 2:
                continue
            short = replace(
                frame, tiles=replace(frame.tiles, tiles=frame.tiles.tiles[1:])
            )
            with self.assertRaisesRegex(
                TsdfError,
                f"observation {frame.sequence}: the block reads the depth "
                r"pixel at row \d+, column \d+, and the evidence holds no "
                "tile for it",
            ):
                _fuse_blocks_from_frames(
                    WORK.room_statement.camera,
                    WORK.room_statement.depth_scale_m,
                    volume.voxel_size_m,
                    volume.truncation_m,
                    volume.block_indices,
                    frames[:position] + (short,) + frames[position + 1:],
                )

    def test_an_obstacle_taken_out_of_the_way_is_refuted(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            path = Path(temporary) / "altered.sftvol"
            path.write_bytes(
                forge(
                    WORK.room_volume_path.read_bytes(),
                    say_the_way_is_clear(ROOM_BLOCKED),
                )
            )
            altered = load_tsdf_block_volume(path)
            # The altered map says the way is clear, and the map tool
            # reading it draws every column of the route free.
            said, absent = measure_tsdf_route(altered, ROOM_BLOCKED)
            self.assertEqual(
                (said, absent),
                (TsdfRouteClearance(98, 1078, 0, 0, None), ()),
            )
            self.assertEqual(
                what_the_map_tool_says(altered, ROOM_BLOCKED),
                (0, 0, None, True),
            )
            statement = state_tsdf_map(altered, WORK.room.session)
            certificate = certify_tsdf_route(
                statement, altered, WORK.room.session, ROOM_BLOCKED
            )
        changed = rows_that_differ(
            WORK.room_volume, altered.tsdf_sums, altered.weights
        )
        check = check_tsdf_route_certificate(statement, certificate)
        self.assertIs(check.verdict, TsdfRouteVerdict.REFUTED)
        self.assertEqual(check.failed_check, "fusion")
        self.assertEqual(check.passed_checks, CHECKS[:-2])
        self.assertIsNone(check.clearance)
        self.assertEqual(
            check.refuted_blocks,
            (
                ((4, -4, -3), 9),
                ((5, -4, -3), 5),
                ((4, -3, -3), 19),
                ((5, -3, -3), 3),
                ((4, -2, -3), 36),
                ((5, -2, -3), 51),
                ((4, -4, -2), 2),
                ((5, -4, -2), 12),
                ((4, -2, -2), 41),
                ((5, -2, -2), 78),
            ),
        )
        # Exactly the blocks that were altered, by exactly the voxels.
        self.assertEqual(
            dict(check.refuted_blocks),
            {
                altered.block_indices[row]: count
                for row, count in changed.items()
            },
        )
        self.assertEqual(
            check.detail,
            "10 of 12 blocks the route touches are not what the scan "
            "gives, the first of them (4, -4, -3) in 9 of 512 voxels",
        )
        # Under the honest statement the altered blocks are no part of
        # the map.
        honest = check_tsdf_route_certificate(
            WORK.room_statement,
            replace(
                certificate,
                statement_sha256=WORK.room_statement.digest_sha256,
            ),
        )
        self.assertIs(honest.verdict, TsdfRouteVerdict.INVALID)
        self.assertEqual(honest.failed_check, "map root")

    def test_a_scan_altered_before_it_was_committed_to_is_refuted(
        self,
    ) -> None:
        volume = WORK.room_volume
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(WORK.room.session_path, temporary)
            drop_a_frame(session_path)
            session = load_scan_session(session_path)
            statement = _statement(
                commit_tsdf_scan(session), commit_tsdf_volume(volume)
            )
            certificate = certify_tsdf_route(
                statement, volume, session, ROOM_CLEAR
            )
            lone = fuse_tsdf_blocks_alone(
                session,
                certificate.block_indices,
                voxel_size_m=volume.voxel_size_m,
                truncation_m=volume.truncation_m,
                frame_stride=volume.frame_stride,
            )
        check = check_tsdf_route_certificate(statement, certificate)
        self.assertIs(check.verdict, TsdfRouteVerdict.REFUTED)
        self.assertEqual(len(check.refuted_blocks), 26)
        self.assertEqual(
            check.refuted_blocks,
            tuple(
                (
                    block.block_index,
                    int(
                        np.count_nonzero(
                            (
                                block.tsdf_sums.view(np.uint64)
                                != certificate.tsdf_sums[place].view(
                                    np.uint64
                                )
                            )
                            | (block.weights != certificate.weights[place])
                        )
                    ),
                )
                for place, block in enumerate(lone)
            ),
        )
        self.assertEqual(check.refuted_blocks[0], ((2, -2, -1), 373))

    def test_a_route_into_space_the_map_does_not_hold_is_not_certified(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TsdfError,
            r"the route touches 3 of 8 blocks that the map does not hold, "
            r"the first of them \(-1, -1, -1\); the map knows nothing of "
            "that space, so the route is not clear, and a certificate "
            "cannot yet show a block is absent",
        ):
            certify_tsdf_route(
                WORK.room_statement,
                WORK.room_volume,
                WORK.room.session,
                ROOM_UNKNOWN,
            )

    def test_a_certificate_is_made_only_under_a_statement_of_these_two(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TsdfError, "not a statement of this volume: session"
        ):
            certify_tsdf_route(
                WORK.statement,
                WORK.room_volume,
                WORK.room.session,
                ROOM_CLEAR,
            )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            session_path = copied(WORK.room.session_path, temporary)
            change_one_sample(session_path)
            other = load_scan_session(session_path)
            with self.assertRaisesRegex(
                TsdfError,
                "not a statement of this volume and this scan: scan root",
            ):
                certify_tsdf_route(
                    WORK.room_statement, WORK.room_volume, other, ROOM_CLEAR
                )
            # Or the scan is rewritten between the two readings of it.
            real = tsdf_route_certificate._gather_pooled_frames

            def gather(session, *arguments, **options):
                return real(other, *arguments, **options)

            with patch(
                "spatialforge.tsdf_route_certificate._gather_pooled_frames",
                gather,
            ):
                with self.assertRaisesRegex(
                    TsdfError,
                    "session inputs changed while the route was certified",
                ):
                    certify_tsdf_route(
                        WORK.room_statement,
                        WORK.room_volume,
                        WORK.room.session,
                        ROOM_CLEAR,
                    )
        for arguments, message in (
            (
                (WORK.statement_path, WORK.volume, WORK.session,
                 FIXTURE_CLEAR),
                "TsdfMapStatement",
            ),
            (
                (WORK.statement, WORK.volume_path, WORK.session,
                 FIXTURE_CLEAR),
                "TsdfBlockVolume",
            ),
            (
                (WORK.statement, WORK.volume, FIXTURE, FIXTURE_CLEAR),
                "loaded ScanSession",
            ),
            (
                (WORK.statement, WORK.volume, WORK.session, ((2, 0), (5, 0))),
                "requires a TsdfRoute",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TsdfError, message):
                    certify_tsdf_route(*arguments)

    def test_certifying_writes_nothing_and_a_check_reads_nothing_else(
        self,
    ) -> None:
        before = tree_snapshot(WORK.root)
        session_before = tree_snapshot(WORK.room.session_path)
        encoded = certify_tsdf_route(
            WORK.room_statement,
            WORK.room_volume,
            WORK.room.session,
            ROOM_CLEAR,
        ).encoded()
        self.assertEqual(encoded, WORK.room_clear.encoded())
        with forbidden_calls(_FORBIDDEN_TO_A_CHECK):
            check = check_tsdf_route_certificate(
                WORK.room_statement, decode_tsdf_route_certificate(encoded)
            )
        self.assertIs(check.verdict, TsdfRouteVerdict.CLEAR)
        self.assertEqual(tree_snapshot(WORK.root), before)
        self.assertEqual(
            tree_snapshot(WORK.room.session_path), session_before
        )
        with self.assertRaisesRegex(TsdfError, "TsdfMapStatement"):
            check_tsdf_route_certificate(
                WORK.statement_path, WORK.room_clear
            )
        with self.assertRaisesRegex(TsdfError, "TsdfRouteCertificate"):
            check_tsdf_route_certificate(WORK.room_statement, encoded)


class AlteredCertificateTests(unittest.TestCase):
    """An honest statement and a certificate that has been tampered with."""

    def stopped(self, certificate, by: str, detail: str | None = None):
        check = check_tsdf_route_certificate(WORK.room_statement, certificate)
        self.assertIs(check.verdict, TsdfRouteVerdict.INVALID)
        self.assertEqual(check.failed_check, by)
        self.assertEqual(check.passed_checks, CHECKS[:CHECKS.index(by)])
        self.assertEqual(check.refuted_blocks, ())
        self.assertIsNone(check.clearance)
        self.assertFalse(check.holds)
        if detail is not None:
            self.assertRegex(check.detail, detail)
        return check

    def test_a_certificate_under_another_statement_or_of_another_shape(
        self,
    ) -> None:
        certificate = WORK.room_clear
        self.stopped(
            replace(certificate, statement_sha256="0" * 64),
            "statement",
            "the certificate was made under statement 0{64}",
        )
        self.stopped(WORK.fixture_clear, "statement")
        self.stopped(
            replace(
                certificate,
                frames=certificate.frames[:-1],
                tile_frames=tuple(
                    tiles
                    for tiles in certificate.tile_frames
                    if tiles.sequence < 19
                ),
            ),
            "shape",
            "the certificate has frames 19 and the statement 20",
        )

    def test_blocks_that_are_not_the_routes(self) -> None:
        certificate = WORK.room_clear
        left_out = certificate.block_indices[7]
        self.stopped(
            without_block(certificate, 7),
            "route",
            f"the route touches block {left_out}, and the certificate does "
            "not hold it".replace("(", r"\(").replace(")", r"\)"),
        )
        # A true block of the map with its true path, which the route
        # does not touch.
        tree = tsdf_volume_merkle_tree(WORK.room_volume)
        row = next(
            row
            for row, block in enumerate(WORK.room_volume.block_indices)
            if block not in certificate.block_indices
        )
        block = WORK.room_volume.block_indices[row]
        entries = sorted(
            list(
                zip(
                    certificate.block_indices,
                    certificate.block_rows,
                    certificate.block_paths,
                    certificate.tsdf_sums,
                    certificate.weights,
                )
            )
            + [
                (
                    block,
                    row,
                    tree.path(row),
                    WORK.room_volume.tsdf_sums[row],
                    WORK.room_volume.weights[row],
                )
            ],
            key=lambda entry: entry[0][::-1],
        )
        self.stopped(
            replace(
                certificate,
                block_indices=tuple(entry[0] for entry in entries),
                block_rows=tuple(entry[1] for entry in entries),
                block_paths=tuple(entry[2] for entry in entries),
                tsdf_sums=frozen(np.stack([entry[3] for entry in entries])),
                weights=frozen(np.stack([entry[4] for entry in entries])),
            ),
            "route",
            f"the certificate holds block {block}, and the route does not "
            "touch it".replace("(", r"\(").replace(")", r"\)"),
        )
        # The route said to run somewhere else altogether.
        self.stopped(
            replace(
                certificate,
                route=replace(ROOM_CLEAR, waypoints=((20, -10), (20, 30))),
            ),
            "route",
        )

    def test_a_route_restated_is_another_route_and_is_answered_truly(
        self,
    ) -> None:
        # A certificate states its route. Stated differently, within
        # the blocks it holds, it is a certificate of that other route,
        # and what it then says is what the map holds for that one.
        certificate = WORK.room_clear
        restated = (
            replace(ROOM_CLEAR, band_first=-4, band_last=4),
            replace(ROOM_CLEAR, min_weight=1),
            replace(ROOM_CLEAR, min_weight=21),
            replace(
                ROOM_CLEAR, waypoints=((20, -10), (50, -10), (50, 19))
            ),
        )
        verdicts = []
        for route in restated:
            with self.subTest(route=route):
                self.assertEqual(
                    route.block_indices(), certificate.block_indices
                )
                check = check_tsdf_route_certificate(
                    WORK.room_statement, replace(certificate, route=route)
                )
                said, absent = measure_tsdf_route(WORK.room_volume, route)
                self.assertEqual(absent, ())
                self.assertEqual(check.clearance, said)
                self.assertIs(
                    check.verdict,
                    TsdfRouteVerdict.CLEAR
                    if said.is_clear
                    else TsdfRouteVerdict.NOT_CLEAR,
                )
                verdicts.append(check.verdict)
        # No voxel was seen twenty-one times by twenty frames.
        self.assertEqual(
            verdicts,
            [
                TsdfRouteVerdict.CLEAR,
                TsdfRouteVerdict.CLEAR,
                TsdfRouteVerdict.NOT_CLEAR,
                TsdfRouteVerdict.CLEAR,
            ],
        )

    def test_frames_blocks_and_tiles_that_are_not_the_statements(
        self,
    ) -> None:
        certificate = WORK.room_blocked
        nudged = list(certificate.frames[4].transform)
        nudged[3] += 0.001
        self.stopped(
            with_frame(certificate, 4, transform=tuple(nudged)),
            "scan root",
            "the frames' leaves lead to ",
        )
        # The voxel that blocks the route, said in the certificate to
        # be empty.
        clearance = measure_tsdf_route(WORK.room_volume, ROOM_BLOCKED)[0]
        self.assertEqual(clearance.occupied_voxels, 14)
        occupied = (
            (certificate.weights >= 3) & (certificate.tsdf_sums <= 0.0)
        )
        place = int(np.argwhere(occupied)[0][0])
        opened = certificate.tsdf_sums.copy()
        opened[occupied] = certificate.weights[occupied]
        self.stopped(
            replace(certificate, tsdf_sums=frozen(opened)),
            "map root",
            f"block {certificate.block_indices[place]} and its path lead "
            "to ".replace("(", r"\(").replace(")", r"\)"),
        )
        self.stopped(
            replace(
                certificate,
                block_rows=(certificate.block_rows[1],)
                + certificate.block_rows[1:],
                block_paths=(certificate.block_paths[1],)
                + certificate.block_paths[1:],
            ),
            "map root",
        )
        tiles = certificate.tile_frames[0]
        first = tiles.tiles.tiles[0]
        changed = first[2].copy()
        changed[3, 3] += 1
        self.stopped(
            with_tile_frame(
                certificate,
                0,
                tiles=replace(
                    tiles.tiles,
                    tiles=((*first[:2], frozen(changed)),)
                    + tiles.tiles.tiles[1:],
                ),
            ),
            "depth roots",
            f"observation {tiles.sequence}: the tile at row {first[0]}, "
            f"column {first[1]} does not lead to the frame's depth root",
        )

    def test_a_certificate_that_holds_less_or_more_than_its_blocks_read(
        self,
    ) -> None:
        certificate = WORK.room_blocked
        position, tiles = next(
            (position, tiles)
            for position, tiles in enumerate(certificate.tile_frames)
            if 2 <= tiles.tile_count < 12
        )
        for left_out in range(tiles.tile_count):
            with self.subTest(tile=left_out):
                self.stopped(
                    with_tile_frame(
                        certificate,
                        position,
                        tiles=replace(
                            tiles.tiles,
                            tiles=tiles.tiles.tiles[:left_out]
                            + tiles.tiles.tiles[left_out + 1:],
                        ),
                        paths=tiles.paths[:left_out]
                        + tiles.paths[left_out + 1:],
                    ),
                    "pixels",
                    f"observation {tiles.sequence}: the block reads the "
                    r"depth pixel at row \d+, column \d+, and the evidence "
                    "holds no tile for it",
                )
        for left_out, frame_tiles in enumerate(certificate.tile_frames):
            with self.subTest(frame=frame_tiles.sequence):
                self.stopped(
                    replace(
                        certificate,
                        tile_frames=certificate.tile_frames[:left_out]
                        + certificate.tile_frames[left_out + 1:],
                    ),
                    "pixels",
                    f"observation {frame_tiles.sequence}: the block reads "
                    "the depth pixel",
                )
        columns = tiles.tiles.tile_columns
        held = {tile[0] * columns + tile[1] for tile in tiles.tiles.tiles}
        spare = next(place for place in range(12) if place not in held)
        tile, path = tile_of_the_scan(
            WORK.room.session_path, tiles.sequence, spare
        )
        entries = sorted(
            list(zip(tiles.tiles.tiles, tiles.paths)) + [(tile, path)],
            key=lambda entry: entry[0][:2],
        )
        self.stopped(
            with_tile_frame(
                certificate,
                position,
                tiles=replace(
                    tiles.tiles, tiles=tuple(entry[0] for entry in entries)
                ),
                paths=tuple(entry[1] for entry in entries),
            ),
            "pixels",
            f"observation {tiles.sequence}: the certificate holds the tile "
            f"at row {tile[0]}, column {tile[1]}, and no block of the "
            "route reads a pixel of it",
        )


class EveryByteTests(unittest.TestCase):
    def test_a_changed_certificate_never_says_what_is_not_so(self) -> None:
        certificate = WORK.fixture_clear
        encoded = certificate.encoded()
        self.assertEqual(len(encoded), FIXTURE_CLEAR_BYTES)
        outcomes: Counter = Counter()
        for position in range(len(encoded)):
            changed = bytearray(encoded)
            changed[position] ^= 1 << (position % 8)
            try:
                decoded = decode_tsdf_route_certificate(bytes(changed))
                check = check_tsdf_route_certificate(WORK.statement, decoded)
            except TsdfError:
                outcomes["not a certificate"] += 1
                continue
            if check.verdict is TsdfRouteVerdict.REFUTED:
                self.fail(f"byte {position} changed and the map was refuted")
            if check.clearance is None:
                self.assertIs(check.verdict, TsdfRouteVerdict.INVALID)
                outcomes[check.failed_check] += 1
                continue
            # It is a certificate of another route. What it says of
            # that route is what the volume says of it.
            self.assertNotEqual(decoded.route, FIXTURE_CLEAR)
            self.assertEqual(
                measure_tsdf_route(WORK.volume, decoded.route),
                (check.clearance, ()),
            )
            outcomes[f"{check.verdict.value}, of another route"] += 1
        self.assertEqual(
            dict(outcomes),
            {
                "not a certificate": 569,
                "statement": 19,
                "shape": 1,
                "route": 22,
                "scan root": 320,
                "map root": 6241,
                "depth roots": 16,
                # The one change that asks for three observations where
                # the route asked for two.
                "not clear, of another route": 1,
            },
        )

    def test_no_certificate_is_a_beginning_of_another(self) -> None:
        encoded = WORK.fixture_clear.encoded()
        for length in range(len(encoded)):
            with self.assertRaises(TsdfError):
                decode_tsdf_route_certificate(encoded[:length])
        with self.assertRaisesRegex(
            TsdfError, "TSDF route certificate holds 1 bytes after its end"
        ):
            decode_tsdf_route_certificate(encoded + b"\x00")
        room = WORK.room_blocked.encoded()
        for length in range(0, len(room), 1009):
            with self.assertRaises(TsdfError):
                decode_tsdf_route_certificate(room[:length])


class CertificateFileTests(unittest.TestCase):
    def test_the_fixtures_certificates_written_out(self) -> None:
        certificate = WORK.fixture_clear
        encoded = certificate.encoded()
        self.assertEqual(certificate.digest_sha256, FIXTURE_CLEAR_SHA256)
        self.assertEqual(encoded[:8], b"SFTRTE01")
        self.assertEqual(struct.unpack_from("<Q", encoded, 8), (463,))
        self.assertEqual(encoded[16:479], FIXTURE_CLEAR_HEADER)
        # The waypoints, the two frames' leaves, then the one block:
        # its row, the block as the map's tree holds it, its path.
        self.assertEqual(encoded[479:511], struct.pack("<4q", 2, 0, 5, 0))
        self.assertEqual(
            encoded[511:865],
            b"".join(frame.leaf_data() for frame in certificate.frames),
        )
        self.assertEqual(encoded[865:873], struct.pack("<Q", 6))
        self.assertEqual(
            encoded[873:893], b"SFBLOCK1" + struct.pack("<3i", 0, 0, 0)
        )
        self.assertEqual(
            encoded[893:893 + 6144],
            WORK.volume.tsdf_sums[6].tobytes()
            + WORK.volume.weights[6].tobytes(),
        )
        self.assertEqual(
            encoded[7037:7133], b"".join(certificate.block_paths[0])
        )
        # And the tiles, as a block's proof holds them.
        self.assertEqual(
            encoded[7133:],
            struct.pack("<QI", 0, 1)
            + struct.pack("<II", 0, 0)
            + struct.pack("<4H", *[1000] * 4)
            + struct.pack("<QI", 1, 1)
            + struct.pack("<II", 0, 0)
            + struct.pack("<4H", *[950] * 4),
        )
        for route, digest, verdict, blocks in (
            (FIXTURE_CLEAR, FIXTURE_CLEAR_SHA256, "clear", 1),
            (FIXTURE_WALL, FIXTURE_WALL_SHA256, "not clear", 2),
            (FIXTURE_FOUR, FIXTURE_FOUR_SHA256, "clear", 4),
        ):
            with self.subTest(route=route):
                made = certify_tsdf_route(
                    WORK.statement, WORK.volume, WORK.session, route
                )
                self.assertEqual(made.digest_sha256, digest)
                self.assertEqual(made.held_block_count, blocks)
                self.assertEqual(
                    check_tsdf_route_certificate(
                        WORK.statement, made
                    ).verdict.value,
                    verdict,
                )

    def test_a_certificate_written_is_the_certificate_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            for name, certificate, statement in (
                ("fixture", WORK.fixture_clear, WORK.statement),
                ("room", WORK.room_clear, WORK.room_statement),
            ):
                with self.subTest(certificate=name):
                    output = Path(temporary) / name / "way.sftroute"
                    report = write_tsdf_route_certificate(
                        certificate, output
                    )
                    self.assertEqual(report.output, output.resolve())
                    self.assertEqual(
                        report.output_bytes, len(certificate.encoded())
                    )
                    self.assertEqual(
                        report.output_digest_sha256,
                        certificate.digest_sha256,
                    )
                    self.assertIs(report.certificate, certificate)
                    self.assertEqual(
                        output.read_bytes(), certificate.encoded()
                    )
                    self.assertEqual(
                        [path.name for path in output.parent.iterdir()],
                        ["way.sftroute"],
                    )
                    loaded = load_tsdf_route_certificate(output)
                    self.assertEqual(loaded.route, certificate.route)
                    self.assertEqual(
                        loaded.encoded(), certificate.encoded()
                    )
                    self.assertIs(
                        check_tsdf_route_certificate(
                            statement, loaded
                        ).verdict,
                        TsdfRouteVerdict.CLEAR,
                    )
            output = Path(temporary) / "fixture" / "way.sftroute"
            with self.assertRaisesRegex(TsdfError, "output already exists"):
                write_tsdf_route_certificate(WORK.fixture_clear, output)
            output.unlink()

            def link(source, target):
                Path(target).write_bytes(b"someone else's")
                raise FileExistsError(target)

            with patch(
                "spatialforge.tsdf_route_certificate.os.link", link
            ):
                with self.assertRaisesRegex(
                    TsdfError, "appeared while writing"
                ):
                    write_tsdf_route_certificate(WORK.fixture_clear, output)
            self.assertEqual(output.read_bytes(), b"someone else's")
            self.assertEqual(
                [path.name for path in output.parent.iterdir()],
                ["way.sftroute"],
            )
            for name in ("way.sftproof", "way.sftroute.bin", "way"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(TsdfError, ".sftroute"):
                        write_tsdf_route_certificate(
                            WORK.fixture_clear, Path(temporary) / name
                        )
            with self.assertRaisesRegex(TsdfError, "TsdfRouteCertificate"):
                write_tsdf_route_certificate(
                    WORK.fixture_clear.encoded(),
                    Path(temporary) / "other.sftroute",
                )
            with self.assertRaisesRegex(TsdfError, "must end in .sftroute"):
                load_tsdf_route_certificate(WORK.statement_path)
            with self.assertRaisesRegex(
                TsdfError, "cannot read TSDF route certificate"
            ):
                load_tsdf_route_certificate(
                    Path(temporary) / "gone.sftroute"
                )

    def test_bytes_that_are_not_exactly_a_certificate_are_refused(
        self,
    ) -> None:
        encoded = WORK.fixture_clear.encoded()
        payload = 16 + 463

        def put(path, value):
            def change(document):
                target = document
                for key in path[:-1]:
                    target = target[key]
                if value is _GONE:
                    del target[path[-1]]
                else:
                    target[path[-1]] = value

            return rebuilt(encoded, change)

        def raw_header(text: str) -> bytes:
            data = text.encode("utf-8")
            return (
                encoded[:8] + struct.pack("<Q", len(data)) + data
                + encoded[payload:]
            )

        text = FIXTURE_CLEAR_HEADER.decode("ascii")
        for held in (bytearray(encoded), memoryview(encoded), "SFTRTE01"):
            with self.subTest(held=type(held).__name__):
                with self.assertRaisesRegex(TsdfError, "decoded from bytes"):
                    decode_tsdf_route_certificate(held)
        cases = {
            "nothing": (b"", "unrecognized signature"),
            "a proof's signature": (
                b"SFTPRF01" + encoded[8:],
                "unrecognized signature",
            ),
            "a header of no length": (
                encoded[:8] + struct.pack("<Q", 0) + encoded[16:],
                "header length is out of range",
            ),
            "a header longer than the file": (
                encoded[:8] + struct.pack("<Q", 16_000) + encoded[16:],
                "TSDF route certificate is truncated inside its header",
            ),
            "a header packed onto one line": (
                raw_header(json.dumps(json.loads(text), sort_keys=True)),
                "not in canonical form",
            ),
            "another schema": (
                put(("schema",), "spatialforge.tsdf-block-proof"),
                "certificate.schema: expected",
            ),
            "another version": (
                put(("schema_version",), "0.2.0"),
                "certificate.schema_version: expected",
            ),
            "a field left out": (
                put(("route", "radius"), _GONE),
                "certificate.route.radius: missing",
            ),
            "a field nobody asked for": (
                put(("verdict",), "clear"),
                "certificate: unknown field 'verdict'",
            ),
            "a route field nobody asked for": (
                put(("route", "clear"), True),
                "certificate.route: unknown field 'clear'",
            ),
            "a statement digest that is not one": (
                put(("statement_sha256",), "abc"),
                "certificate.statement_sha256: expected a SHA-256",
            ),
            "an up axis there is not": (
                put(("route", "up_axis"), "w"),
                "certificate.route.up_axis: expected x, y or z",
            ),
            "an up axis by number": (
                put(("route", "up_axis"), 2),
                "certificate.route.up_axis: expected x, y or z",
            ),
            "a band upside down": (
                put(("route", "band_last"), -1),
                "at least one voxel",
            ),
            "a band in metres": (
                put(("route", "band_first"), 0.0),
                "certificate.route.band_first",
            ),
            "a radius below nothing": (
                put(("route", "radius"), -1),
                "certificate.route.radius",
            ),
            "a radius too wide": (
                put(("route", "radius"), 65),
                "radius is a count of voxels",
            ),
            "no observations asked": (
                put(("route", "min_weight"), 0),
                "certificate.route.min_weight",
            ),
            "no waypoints": (
                put(("route", "waypoints"), 0),
                "certificate.route.waypoints",
            ),
            "one waypoint more than it holds": (
                put(("route", "waypoints"), 3),
                "waypoint is two voxel indices within",
            ),
            "one waypoint fewer than it holds": (
                put(("route", "waypoints"), 1),
                "a TSDF frame leaf starts with its tag",
            ),
            "more waypoints than there are bytes for": (
                put(("route", "waypoints"), 10**9),
                "TSDF route certificate counts more than its bytes can hold",
            ),
            "no frames": (put(("frames",), 0), "certificate.frames"),
            "no blocks held": (
                put(("blocks", "held"), 0),
                "certificate.blocks.held",
            ),
            "more blocks than there are bytes for": (
                put(("blocks", "held"), 70_000),
                "counts more than its bytes can hold",
            ),
            "one block more than it holds": (
                put(("blocks", "held"), 2),
                "counts more than its bytes can hold",
            ),
            "a map of no blocks": (
                put(("blocks", "of"), 0),
                "certificate.blocks.of",
            ),
            "a row past the map's last": (
                put(("blocks", "of"), 6),
                "row lies outside the blocks it counts",
            ),
            "tiles too large": (
                put(("image", "tile_pixels"), 1025),
                "tile size",
            ),
            "one tile frame fewer than it holds": (
                put(("tile_frames",), 1),
                "TSDF route certificate holds 28 bytes after its end",
            ),
            "an array": (
                raw_header(text.replace('"frames": 2', '"frames": [2]')),
                "holds an array",
            ),
            "a key twice": (
                raw_header(
                    text.replace(
                        '  "tile_frames": 2',
                        '  "tile_frames": 2,\n  "tile_frames": 2',
                    )
                ),
                "duplicate JSON key 'tile_frames'",
            ),
            "not a number": (
                raw_header(text.replace('"frames": 2', '"frames": NaN')),
                "invalid JSON",
            ),
            "not ascii": (
                raw_header(text.replace("0.1.0", "0.1." + chr(233))),
                "must be ASCII JSON",
            ),
            "the same waypoint twice": (
                encoded[:payload]
                + struct.pack("<4q", 2, 0, 2, 0)
                + encoded[payload + 32:],
                "must differ from the one before it",
            ),
            "the block's tag damaged": (
                encoded[:873] + b"X" + encoded[874:],
                "block does not start with its tag",
            ),
            "a frame named and no tile of it": (
                encoded[:-28] + struct.pack("<QI", 1, 0) + encoded[-16:],
                "TSDF route certificate names a frame and holds no tile",
            ),
        }
        for name, (changed, message) in cases.items():
            with self.subTest(case=name):
                self.assertNotEqual(changed, encoded)
                with self.assertRaisesRegex(TsdfError, message):
                    decode_tsdf_route_certificate(changed)

    def test_certificates_validate_themselves(self) -> None:
        certificate = WORK.room_clear
        held = certificate.held_block_count
        cases = {
            "a statement digest that is not one": (
                {"statement_sha256": "abc"},
                "statement digest",
            ),
            "tiles of no pixels": ({"tile_pixels": 0}, "tile size"),
            "no frames": ({"frames": ()}, "at least one"),
            "frames out of order": (
                {"frames": certificate.frames[::-1]},
                "every frame, in replay order",
            ),
            "a route that is not one": (
                {"route": ROOM_CLEAR.waypoints},
                "needs a TsdfRoute",
            ),
            "a map of no blocks": ({"block_count": 0}, "block count"),
            "no blocks": ({"block_indices": ()}, "from 1 to 65536 blocks"),
            "blocks as a list": (
                {"block_indices": list(certificate.block_indices)},
                "from 1 to 65536 blocks",
            ),
            "blocks out of order": (
                {"block_indices": certificate.block_indices[::-1]},
                "distinct and in canonical order",
            ),
            "a block twice": (
                {
                    "block_indices": (certificate.block_indices[0],) * held
                },
                "distinct and in canonical order",
            ),
            "a row too few": (
                {"block_rows": certificate.block_rows[:-1]},
                "a row of the map for each block",
            ),
            "a row past the map's last": (
                {"block_rows": (524,) + certificate.block_rows[1:]},
                "a row of the map for each block",
            ),
            "sums that can be written": (
                {"tsdf_sums": certificate.tsdf_sums.copy()},
                "immutable float64",
            ),
            "sums for fewer blocks": (
                {"tsdf_sums": frozen(certificate.tsdf_sums[:-1])},
                "immutable float64",
            ),
            "weights of another kind": (
                {"weights": frozen(certificate.weights.astype(np.int64))},
                "immutable uint32",
            ),
            "a path too few": (
                {"block_paths": certificate.block_paths[:-1]},
                "a path for each block",
            ),
            "a path one step short": (
                {
                    "block_paths": (certificate.block_paths[0][:-1],)
                    + certificate.block_paths[1:]
                },
                r"block path here is \d+ hashes",
            ),
            "tiles that are not tiles": (
                {"tile_frames": (certificate.tile_frames[0].tiles,)},
                "must be TsdfProofFrameTiles",
            ),
            "tile frames out of order": (
                {"tile_frames": certificate.tile_frames[::-1]},
                "increasing frame order",
            ),
            "tiles of a frame it does not hold": (
                {
                    "tile_frames": (
                        replace(certificate.tile_frames[0], sequence=20),
                    )
                },
                "a frame it does not hold",
            ),
            "tiles cut on another grid": (
                {"tile_pixels": 8},
                "not cut on its tile grid",
            ),
        }
        for name, (changes, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(TsdfError, message):
                    replace(certificate, **changes)
        self.assertIsInstance(
            certificate.tile_frames[0], TsdfProofFrameTiles
        )


class RouteCliTests(unittest.TestCase):
    def run_cli(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(["reconstruct", *arguments])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def certify(self, output: Path, *route: str, volume=None):
        return self.run_cli(
            "tsdf-route-certify",
            str(WORK.statement_path),
            str(volume or WORK.volume_path),
            str(FIXTURE),
            str(output),
            *route,
        )

    def test_cli_certifies_a_clear_route_and_checks_it_alone(self) -> None:
        before = tree_snapshot(WORK.root)
        described = (
            "route: up=z band=0..0 radius=0 min_weight=2 waypoints=2\n"
            "route_metres: length=0.375 width=0.125 from=0.000 to=0.125\n"
            "certificate_holds: blocks=1 frames=2 frames_with_tiles=2 "
            "tiles=2 tile_pixels=16 depth_bytes=16 tile_path_hashes=0 "
            "block_path_hashes=3\n"
        )
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "way.sftroute"
            route = (
                "--band", "0", "0", "--min-weight", "2",
                "--through", "2", "0", "--through", "5", "0",
            )
            code, printed, errors = self.certify(output, *route)
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF ROUTE CERTIFY scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "volume: verified\n"
                "statement_recomputed: matched\n"
                + described
                + "map_says: clear unobserved_voxels=0 occupied_voxels=0\n"
                "certificate_bytes: 7189\n"
                f"certificate_sha256: {FIXTURE_CLEAR_SHA256}\n"
                f"output: {output.resolve()}\n"
                "statement_modified: no\n"
                "volume_modified: no\n"
                "session_modified: no\n",
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                FIXTURE_CLEAR_SHA256,
            )
            code, printed, errors = self.certify(output, *route)
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF ROUTE CERTIFY FAILED {WORK.statement_path}\n"
                f"- output already exists: {output.resolve()}\n",
            )

            alone = Path(temporary) / "alone"
            alone.mkdir()
            statement = alone / "map.sftcommit"
            certificate = alone / "way.sftroute"
            shutil.copyfile(WORK.statement_path, statement)
            shutil.copyfile(output, certificate)
            with forbidden_calls(
                (
                    "spatialforge.cli.load_scan_session",
                    "spatialforge.cli.load_tsdf_block_volume",
                )
            ):
                code, printed, errors = self.run_cli(
                    "tsdf-route-check", str(statement), str(certificate)
                )
            self.assertEqual(errors, "")
            self.assertEqual(code, 0)
            self.assertEqual(
                printed,
                "TSDF ROUTE CHECK scan-synthetic-0001\n"
                "statement: valid\n"
                f"statement_sha256: {FIXTURE_STATEMENT_SHA256}\n"
                "certificate: well formed\n"
                f"certificate_sha256: {FIXTURE_CLEAR_SHA256}\n"
                + described
                + "".join(f"check {name}: passed\n" for name in CHECKS)
                + "fusion: blocks=1 refuted=0 selected=2 ready=2\n"
                "clearance: columns=4 voxels=4 unobserved=0 occupied=0\n"
                "session_read: no\n"
                "volume_read: no\n"
                "statement_modified: no\n"
                "certificate_modified: no\n"
                "verdict: clear\n"
                "result: every block the route touches is what the scan "
                "gives, and the route is clear\n",
            )
        self.assertEqual(tree_snapshot(WORK.root), before)

    def test_cli_says_a_route_into_a_wall_is_not_clear(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            output = Path(temporary) / "wall.sftroute"
            code, printed, errors = self.certify(
                output,
                "--band", "0", "0", "--min-weight", "2",
                "--through", "5", "0", "--through", "10", "0",
            )
            self.assertEqual((code, errors), (0, ""))
            self.assertIn(
                "map_says: not clear unobserved_voxels=0 occupied_voxels=3\n",
                printed,
            )
            self.assertIn(
                f"certificate_sha256: {FIXTURE_WALL_SHA256}\n", printed
            )
            code, printed, errors = self.run_cli(
                "tsdf-route-check", str(WORK.statement_path), str(output)
            )
        self.assertEqual(errors, "")
        self.assertEqual(code, 1)
        self.assertEqual(
            printed.splitlines()[8:],
            [f"check {name}: passed" for name in CHECKS[:-1]]
            + [
                "check clearance: failed",
                "detail: 0 of 6 voxels of the route were observed fewer "
                "than 2 times and 3 are at or behind a surface, the first "
                "of them voxel (8, 0, 0)",
                "fusion: blocks=2 refuted=0 selected=2 ready=2",
                "clearance: columns=6 voxels=6 unobserved=0 occupied=3",
                "session_read: no",
                "volume_read: no",
                "statement_modified: no",
                "certificate_modified: no",
                "verdict: not clear",
                "result: every block the route touches is what the scan "
                "gives, and the route is not clear",
            ],
        )

    def test_cli_refutes_a_route_through_an_altered_map(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            statement = Path(temporary) / "forged.sftcommit"
            output = Path(temporary) / "way.sftroute"
            code, _, errors = self.run_cli(
                "tsdf-map-commit",
                str(WORK.forged_path), str(FIXTURE), str(statement),
            )
            self.assertEqual((code, errors), (0, ""))
            code, printed, errors = self.run_cli(
                "tsdf-route-certify",
                str(statement), str(WORK.forged_path), str(FIXTURE),
                str(output),
                "--band", "-8", "-1", "--min-weight", "1",
                "--through", "3", "-5", "--through", "6", "-2",
            )
            self.assertEqual((code, errors), (0, ""))
            code, printed, errors = self.run_cli(
                "tsdf-route-check", str(statement), str(output)
            )
            self.assertEqual((code, errors), (1, ""))
            lines = printed.splitlines()
            self.assertEqual(
                lines[8:20],
                [f"check {name}: passed" for name in CHECKS[:-2]]
                + [
                    "check fusion: failed",
                    "check clearance: not reached",
                    "detail: 1 of 1 blocks the route touches are not what "
                    "the scan gives, the first of them (0, -1, -1) in 1 of "
                    "512 voxels",
                    "fusion: blocks=1 refuted=1 selected=2 ready=2",
                ],
            )
            self.assertEqual(
                lines[-2:],
                [
                    "verdict: refuted",
                    "result: where the route runs, the map is not what the "
                    "scan gives",
                ],
            )
            # Against the true map's statement it shows nothing.
            code, printed, errors = self.run_cli(
                "tsdf-route-check", str(WORK.statement_path), str(output)
            )
            self.assertEqual((code, errors), (1, ""))
            self.assertEqual(
                printed.splitlines()[-2:],
                [
                    "verdict: invalid",
                    "result: the certificate shows nothing about the map "
                    "or the route; it failed the check of its statement",
                ],
            )

    def test_cli_refuses_what_it_cannot_certify_or_read(self) -> None:
        before = tree_snapshot(WORK.root)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            folder = Path(temporary)
            output = folder / "way.sftroute"
            cases = (
                (
                    "space the map does not hold",
                    (str(output), "--band", "20", "20",
                     "--through", "0", "0"),
                    "- the route touches 1 of 1 blocks that the map does "
                    "not hold, the first of them (0, 0, 2); the map knows "
                    "nothing of that space, so the route is not clear, and "
                    "a certificate cannot yet show a block is absent\n",
                ),
                (
                    "a band upside down",
                    (str(output), "--band", "3", "0", "--through", "0", "0"),
                    "- a TSDF route's band must hold at least one voxel\n",
                ),
                (
                    "a radius below nothing",
                    (str(output), "--band", "0", "0", "--radius", "-1",
                     "--through", "0", "0"),
                    "- a TSDF route's radius is a count of voxels from 0 "
                    "to 64\n",
                ),
                (
                    "the same column twice running",
                    (str(output), "--band", "0", "0",
                     "--through", "2", "0", "--through", "2", "0"),
                    "- a TSDF route's waypoint must differ from the one "
                    "before it\n",
                ),
                (
                    "an output of another kind",
                    (str(folder / "way.bin"), "--band", "0", "0",
                     "--through", "2", "0"),
                    "- output filename must end in .sftroute\n",
                ),
            )
            for name, arguments, message in cases:
                with self.subTest(certify=name):
                    code, printed, errors = self.run_cli(
                        "tsdf-route-certify",
                        str(WORK.statement_path), str(WORK.volume_path),
                        str(FIXTURE), *arguments,
                    )
                    self.assertEqual((code, printed), (2, ""))
                    self.assertEqual(
                        errors,
                        f"TSDF ROUTE CERTIFY FAILED {WORK.statement_path}\n"
                        + message,
                    )
            for missing in (("--band", "0", "0"), ("--through", "0", "0")):
                with self.subTest(given=missing):
                    code, printed, errors = self.certify(output, *missing)
                    self.assertEqual((code, printed), (2, ""))
                    self.assertIn("required", errors)
            code, printed, errors = self.certify(
                output, "--band", "0", "0", "--up-axis", "w",
                "--through", "0", "0",
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertIn("--up-axis", errors)
            self.assertEqual(list(folder.iterdir()), [])

            encoded = WORK.fixture_clear.encoded()
            (folder / "cut.sftroute").write_bytes(encoded[:-1])
            code, printed, errors = self.run_cli(
                "tsdf-route-check",
                str(WORK.statement_path), str(folder / "cut.sftroute"),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF ROUTE CHECK FAILED {folder / 'cut.sftroute'}\n"
                "- TSDF route certificate is truncated inside a tile's "
                "samples\n",
            )
            code, printed, errors = self.run_cli(
                "tsdf-route-check",
                str(WORK.statement_path), str(WORK.statement_path),
            )
            self.assertEqual((code, printed), (2, ""))
            self.assertEqual(
                errors,
                f"TSDF ROUTE CHECK FAILED {WORK.statement_path}\n"
                "- certificate filename must end in .sftroute\n",
            )
        self.assertEqual(tree_snapshot(WORK.root), before)


if __name__ == "__main__":
    unittest.main()
