"""The analysis and rendering tools, on fixtures rather than real datasets.

`tools/` produces the numbers and the pictures that get published, so the
parts that can silently lie need pinning: that held-out evaluation really
excludes the fused frames, that the manifest records what was actually run,
and that the renderer draws the points it was given.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_plan import plan_tsdf_blocks

from tools.render_point_cloud import main as render_main
from tools.render_point_cloud import read_xyz_ply
from tools.tum_reconstruction_report import main as report_main

TEST_ROOT = Path(__file__).resolve().parent
FIXTURE = TEST_ROOT / "fixtures" / "minimal.vgsession"

# The fixture's surfaces sit exactly on voxel boundaries, so the held-out
# frame's depth lands exactly on the reconstructed zero level set. Trilinear
# interpolation recovers that exactly; nearest-voxel sampling is off by
# exactly half a voxel, which is the grid quantisation floor.
VOXEL_M = 0.125
EXACT_TRILINEAR_MM = 0.0
EXACT_NEAREST_MM = -62.5


def build_plan(parent: Path, *, frame_stride: int) -> Path:
    output = parent / f"stride{frame_stride}.sftplan"
    plan_tsdf_blocks(
        load_scan_session(FIXTURE),
        output,
        voxel_size_m=VOXEL_M,
        truncation_m=0.5,
        frame_stride=frame_stride,
    )
    return output


def run_report(arguments: list[str]) -> str:
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = report_main(arguments)
    if exit_code != 0:
        raise AssertionError(f"report exited {exit_code}")
    return stdout.getvalue()


def write_ply(path: Path, points: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property double x",
        "property double y",
        "property double z",
        "end_header",
    ]
    lines.extend(
        f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def sample_cloud() -> np.ndarray:
    """A deliberately asymmetric cloud.

    A shape with rotational symmetry would render identically from several
    orbit positions, and the GIF encoder collapses identical frames — which
    would hide an orbit that never actually moved.
    """

    grid = np.stack(
        np.meshgrid(
            np.linspace(-1.0, 1.0, 12),
            np.linspace(-0.6, 0.9, 10),
            np.linspace(-0.4, 0.5, 6),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    spur = np.stack(
        [
            np.linspace(1.0, 1.8, 20),
            np.full(20, -0.55),
            np.linspace(-0.4, 0.2, 20),
        ],
        axis=-1,
    )
    return np.concatenate([grid, spur])


class HeldOutReportTests(unittest.TestCase):
    def test_reports_the_fixture_residual_exactly(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = build_plan(temporary_root, frame_stride=2)
            manifest_path = temporary_root / "result.json"
            output = run_report(
                [
                    str(FIXTURE),
                    str(plan_path),
                    "--held-out-offset",
                    "1",
                    "--pixel-step",
                    "1",
                    "--manifest-out",
                    str(manifest_path),
                ]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIn("held-out frames: 1 (never fused)", output)
        self.assertIn("inside observed voxels: 4 (100.0%)", output)

        evaluation = manifest["evaluation"]
        self.assertEqual(evaluation["held_out_frames"], 1)
        self.assertEqual(evaluation["held_out_offset"], 1)
        self.assertEqual(evaluation["depth_samples"], 4)
        self.assertEqual(evaluation["samples_in_observed_voxels"], 4)
        self.assertEqual(evaluation["coverage_fraction"], 1.0)
        self.assertAlmostEqual(
            evaluation["trilinear"]["median_absolute_mm"],
            EXACT_TRILINEAR_MM,
            places=9,
        )
        self.assertAlmostEqual(
            evaluation["nearest_voxel"]["mean_signed_mm"],
            EXACT_NEAREST_MM,
            places=9,
        )

    def test_manifest_records_what_was_actually_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = build_plan(temporary_root, frame_stride=2)
            manifest_path = temporary_root / "result.json"
            run_report(
                [
                    str(FIXTURE),
                    str(plan_path),
                    "--held-out-offset",
                    "1",
                    "--pixel-step",
                    "1",
                    "--manifest-out",
                    str(manifest_path),
                ]
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema"], "spatialforge.result-manifest")
        self.assertEqual(manifest["measurement"], "held-out-tsdf-residual")
        self.assertIn(
            "not distance to a surveyed surface",
            manifest["measures"],
        )

        reconstruction = manifest["reconstruction"]
        self.assertEqual(reconstruction["path"], "sparse-block-vector-fusion")
        self.assertIs(reconstruction["persisted"], False)
        self.assertEqual(reconstruction["voxel_size_m"], VOXEL_M)
        self.assertEqual(reconstruction["frame_stride"], 2)
        self.assertEqual(reconstruction["fused_frames"], 1)
        self.assertEqual(
            manifest["inputs"]["session_id"],
            "scan-synthetic-0001",
        )
        self.assertEqual(len(manifest["inputs"]["plan_sha256"]), 64)
        self.assertEqual(manifest["environment"]["numpy"], np.__version__)

    def test_evaluation_set_cannot_become_the_fused_set(self) -> None:
        """The whole result is meaningless if these two sets overlap."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = build_plan(temporary_root, frame_stride=2)
            for offset in (0, 2, 4):
                with self.subTest(offset=offset):
                    with self.assertRaises(SystemExit) as caught:
                        report_main(
                            [
                                str(FIXTURE),
                                str(plan_path),
                                "--held-out-offset",
                                str(offset),
                            ]
                        )
                    self.assertIn(
                        "would be the fused ones",
                        str(caught.exception),
                    )

    def test_no_manifest_is_written_unless_asked(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            plan_path = build_plan(temporary_root, frame_stride=2)
            before = sorted(path.name for path in temporary_root.iterdir())
            run_report(
                [
                    str(FIXTURE),
                    str(plan_path),
                    "--held-out-offset",
                    "1",
                    "--pixel-step",
                    "1",
                ]
            )
            after = sorted(path.name for path in temporary_root.iterdir())

        self.assertEqual(before, after)


class PointCloudRenderTests(unittest.TestCase):
    def test_renders_a_still_and_an_orbit(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.ply"
            write_ply(ply_path, sample_cloud())
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = render_main(
                    [
                        str(ply_path),
                        str(temporary_root / "out"),
                        "--size",
                        "96",
                        "--frames",
                        "4",
                    ]
                )
            still_path = temporary_root / "out.png"
            gif_path = temporary_root / "out.gif"
            self.assertEqual(exit_code, 0)
            self.assertTrue(still_path.exists())
            self.assertTrue(gif_path.exists())

            with Image.open(still_path) as still:
                self.assertEqual(still.size, (96, 96))
                pixels = np.asarray(still.convert("RGB"))
            with Image.open(gif_path) as gif:
                self.assertEqual(gif.size, (96, 96))
                self.assertEqual(gif.n_frames, 4)

        # The cloud must actually be drawn, not left as bare background.
        self.assertGreater(len(np.unique(pixels.reshape(-1, 3), axis=0)), 8)

    def test_rendering_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.ply"
            write_ply(ply_path, sample_cloud())
            renders = []
            for index in range(2):
                with redirect_stdout(io.StringIO()):
                    render_main(
                        [
                            str(ply_path),
                            str(temporary_root / f"run{index}"),
                            "--size",
                            "96",
                            "--frames",
                            "3",
                        ]
                    )
                renders.append(
                    (
                        (temporary_root / f"run{index}.png").read_bytes(),
                        (temporary_root / f"run{index}.gif").read_bytes(),
                    )
                )

        self.assertEqual(renders[0], renders[1])

    def test_ply_reader_rejects_unusable_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            headerless = temporary_root / "headerless.ply"
            headerless.write_text("1.0 2.0 3.0\n", encoding="ascii")
            with self.assertRaises(SystemExit):
                read_xyz_ply(headerless)

            empty = temporary_root / "empty.ply"
            empty.write_text(
                "ply\nformat ascii 1.0\nend_header\n",
                encoding="ascii",
            )
            with self.assertRaises(SystemExit):
                read_xyz_ply(empty)

    def test_reader_round_trips_the_points_it_was_given(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            path = Path(temporary_dir) / "cloud.ply"
            points = sample_cloud()
            write_ply(path, points)
            loaded = read_xyz_ply(path)

        self.assertEqual(loaded.shape, points.shape)
        np.testing.assert_allclose(loaded, points, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
