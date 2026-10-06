"""The analysis and rendering tools, on fixtures rather than real datasets.

`tools/` produces the numbers and the pictures that get published, so the
parts that can silently lie need pinning: that held-out evaluation really
excludes the fused frames, that the manifest records what was actually run,
and that the renderer draws the points it was given.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spatialforge import (
    allocate_empty_tsdf_blocks,
    fuse_tsdf_plan_streaming,
    load_tsdf_block_plan,
    write_tsdf_block_volume,
)
from spatialforge.session_loader import load_scan_session
from spatialforge.tsdf_block_mesh import extract_tsdf_block_mesh
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


def build_volume(
    parent: Path,
    *,
    frame_stride: int,
    voxel_size_m: float = VOXEL_M,
) -> Path:
    """Plan, fuse and persist the fixture, returning the ``.sftvol``."""

    stem = f"stride{frame_stride}-{voxel_size_m}"
    plan_path = parent / f"{stem}.sftplan"
    plan_tsdf_blocks(
        load_scan_session(FIXTURE),
        plan_path,
        voxel_size_m=voxel_size_m,
        truncation_m=0.5,
        frame_stride=frame_stride,
    )
    plan = load_tsdf_block_plan(plan_path)
    session = load_scan_session(FIXTURE)
    storage = allocate_empty_tsdf_blocks(plan, session)
    receipt = fuse_tsdf_plan_streaming(storage, session)
    output = parent / f"{stem}.sftvol"
    write_tsdf_block_volume(storage, receipt, output)
    return output


@contextmanager
def source_state(commit: str | None, clean: bool | None):
    """Pin the recorded source state so tests do not read the real repo.

    Whether this checkout happens to be dirty is not a property of the tool,
    and a test that depends on it fails for unrelated reasons.
    """

    with patch(
        "tools.tum_reconstruction_report.source_state",
        return_value=(commit, clean),
    ):
        yield


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
            volume_path = build_volume(temporary_root, frame_stride=2)
            manifest_path = temporary_root / "result.json"
            with source_state("abc123", True):
                output = run_report(
                    [
                        str(FIXTURE),
                        str(volume_path),
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
            volume_path = build_volume(temporary_root, frame_stride=2)
            manifest_path = temporary_root / "result.json"
            with source_state("abc123", True):
                run_report(
                    [
                        str(FIXTURE),
                        str(volume_path),
                        "--held-out-offset",
                        "1",
                        "--pixel-step",
                        "1",
                        "--manifest-out",
                        str(manifest_path),
                    ]
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_commit"], "abc123")
        self.assertIs(manifest["source_worktree_clean"], True)
        self.assertEqual(manifest["schema"], "spatialforge.result-manifest")
        self.assertEqual(manifest["measurement"], "held-out-tsdf-residual")
        self.assertIn(
            "not distance to a surveyed surface",
            manifest["measures"],
        )

        reconstruction = manifest["reconstruction"]
        self.assertEqual(
            reconstruction["path"],
            "sparse-block-streaming-fusion",
        )
        self.assertIs(reconstruction["persisted"], True)
        self.assertEqual(reconstruction["voxel_size_m"], VOXEL_M)
        self.assertEqual(reconstruction["frame_stride"], 2)
        self.assertEqual(reconstruction["fused_frames"], 1)
        self.assertEqual(
            manifest["inputs"]["session_id"],
            "scan-synthetic-0001",
        )
        self.assertEqual(len(manifest["inputs"]["plan_sha256"]), 64)
        self.assertEqual(len(manifest["inputs"]["volume_sha256"]), 64)
        self.assertIsNone(manifest["mesh"])
        self.assertEqual(manifest["schema_version"], "0.2.0")
        self.assertEqual(manifest["environment"]["numpy"], np.__version__)

    def test_evaluation_set_cannot_become_the_fused_set(self) -> None:
        """The whole result is meaningless if these two sets overlap."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            for offset in (0, 2, 4):
                with self.subTest(offset=offset):
                    with self.assertRaises(SystemExit) as caught:
                        report_main(
                            [
                                str(FIXTURE),
                                str(volume_path),
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
            volume_path = build_volume(temporary_root, frame_stride=2)
            before = sorted(path.name for path in temporary_root.iterdir())
            run_report(
                [
                    str(FIXTURE),
                    str(volume_path),
                    "--held-out-offset",
                    "1",
                    "--pixel-step",
                    "1",
                ]
            )
            after = sorted(path.name for path in temporary_root.iterdir())

        self.assertEqual(before, after)


class ArtifactChainTests(unittest.TestCase):
    """The numbers, the volume and the picture must be one reconstruction."""

    def report(self, arguments: list[str]) -> dict:
        manifest_path = Path(arguments[-1])
        with source_state("abc123", True):
            run_report(arguments)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_a_mesh_of_the_scored_volume_is_recorded_by_digest(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            mesh_path = temporary_root / "mesh.ply"
            mesh_report = extract_tsdf_block_mesh(volume_path, mesh_path)
            manifest = self.report(
                [
                    str(FIXTURE),
                    str(volume_path),
                    "--mesh",
                    str(mesh_path),
                    "--pixel-step",
                    "1",
                    "--manifest-out",
                    str(temporary_root / "result.json"),
                ]
            )

        mesh = manifest["mesh"]
        self.assertEqual(mesh["sha256"], mesh_report.output_digest_sha256)
        self.assertEqual(
            mesh["source_volume_sha256"],
            manifest["inputs"]["volume_sha256"],
        )
        self.assertEqual(mesh["triangles"], mesh_report.triangles_written)
        self.assertEqual(mesh["vertices"], mesh_report.vertices_written)
        self.assertEqual(mesh["minimum_weight"], 1)
        self.assertEqual(mesh["minimum_component_triangles"], 1)
        # With no offset given, half the stride is held out.
        self.assertEqual(manifest["evaluation"]["held_out_offset"], 1)

    def test_a_mesh_of_some_other_volume_is_refused(self) -> None:
        """The exact mistake this chain exists to make impossible."""

        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            other_volume = build_volume(
                temporary_root,
                frame_stride=2,
                voxel_size_m=0.25,
            )
            other_mesh = temporary_root / "other.ply"
            extract_tsdf_block_mesh(other_volume, other_mesh)
            manifest_path = temporary_root / "result.json"
            with self.assertRaises(SystemExit) as caught, source_state(
                "abc123", True
            ):
                report_main(
                    [
                        str(FIXTURE),
                        str(volume_path),
                        "--mesh",
                        str(other_mesh),
                        "--manifest-out",
                        str(manifest_path),
                    ]
                )
            written = manifest_path.exists()

        self.assertIn(
            "was not extracted from this volume",
            str(caught.exception),
        )
        self.assertFalse(written)

    def test_a_volume_fused_from_every_frame_cannot_be_scored(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            volume_path = build_volume(Path(temporary_dir), frame_stride=1)
            with self.assertRaises(SystemExit) as caught:
                report_main([str(FIXTURE), str(volume_path)])

        self.assertIn("no held-out frames", str(caught.exception))

    def test_a_volume_is_only_scored_against_its_own_scan(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            altered = temporary_root / "altered.vgsession"
            shutil.copytree(FIXTURE, altered)
            depth = altered / "data" / "depth" / "000001.pgm"
            original = depth.read_bytes()
            changed = original.replace(b"950", b"900")
            self.assertNotEqual(changed, original)
            depth.write_bytes(changed)
            with self.assertRaises(SystemExit) as caught:
                report_main([str(altered), str(volume_path)])

        self.assertIn("replay digest does not match", str(caught.exception))


class SourceProvenanceTests(unittest.TestCase):
    """A manifest that cannot name its own source code is not evidence."""

    def report_with(
        self,
        temporary_root: Path,
        commit: str | None,
        clean: bool | None,
        *extra: str,
    ) -> Path:
        volume_path = build_volume(temporary_root, frame_stride=2)
        manifest_path = temporary_root / "result.json"
        with source_state(commit, clean):
            run_report(
                [
                    str(FIXTURE),
                    str(volume_path),
                    "--held-out-offset",
                    "1",
                    "--pixel-step",
                    "1",
                    "--manifest-out",
                    str(manifest_path),
                    *extra,
                ]
            )
        return manifest_path

    def test_a_dirty_tree_is_refused_by_default(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            with self.assertRaises(SystemExit) as caught:
                self.report_with(temporary_root, "abc123", False)
            self.assertIn("dirty working tree", str(caught.exception))
            self.assertFalse((temporary_root / "result.json").exists())

    def test_a_dirty_tree_is_recorded_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            manifest_path = self.report_with(
                Path(temporary_dir),
                "abc123",
                False,
                "--allow-dirty",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["source_commit"], "abc123")
        self.assertIs(manifest["source_worktree_clean"], False)

    def test_an_unavailable_repository_is_recorded_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            manifest_path = self.report_with(
                Path(temporary_dir),
                None,
                None,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertIsNone(manifest["source_commit"])
        self.assertIsNone(manifest["source_worktree_clean"])


class OutputSafetyTests(unittest.TestCase):
    """Outputs must never overwrite anything, least of all their inputs."""

    def test_report_refuses_to_overwrite_an_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            manifest_path = temporary_root / "result.json"
            manifest_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught, source_state(
                "abc123", True
            ):
                report_main(
                    [
                        str(FIXTURE),
                        str(volume_path),
                        "--held-out-offset",
                        "1",
                        "--manifest-out",
                        str(manifest_path),
                    ]
                )
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                "{}",
            )

    def test_report_refuses_to_write_over_its_own_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            volume_path = build_volume(temporary_root, frame_stride=2)
            original = volume_path.read_bytes()
            # The plan and the session directory itself, plus a file inside
            # the session: all three are inputs and none may be written over.
            for target, expected in (
                (volume_path, "must end in .json"),
                (FIXTURE, "must end in .json"),
                (FIXTURE / "manifest.json", "write over an input"),
            ):
                with self.subTest(target=target.name):
                    with self.assertRaises(SystemExit) as caught, source_state(
                        "abc123", True
                    ):
                        report_main(
                            [
                                str(FIXTURE),
                                str(volume_path),
                                "--held-out-offset",
                                "1",
                                "--manifest-out",
                                str(target),
                            ]
                        )
                    self.assertIn(expected, str(caught.exception))
            self.assertEqual(volume_path.read_bytes(), original)
            self.assertTrue((FIXTURE / "manifest.json").exists())

    def test_renderer_refuses_to_overwrite_existing_images(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.ply"
            write_ply(ply_path, sample_cloud())
            existing = temporary_root / "out.png"
            existing.write_bytes(b"not an image")
            with self.assertRaises(SystemExit) as caught:
                render_main(
                    [
                        str(ply_path),
                        str(temporary_root / "out"),
                        "--size",
                        "32",
                    ]
                )
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual(existing.read_bytes(), b"not an image")
            self.assertFalse((temporary_root / "out.gif").exists())

    def test_renderer_refuses_to_write_over_its_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.png"
            write_ply(ply_path, sample_cloud())
            with self.assertRaises(SystemExit) as caught:
                render_main(
                    [
                        str(ply_path),
                        str(temporary_root / "cloud"),
                        "--size",
                        "32",
                    ]
                )
            self.assertIn("write over an input", str(caught.exception))

    def test_a_failed_render_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.ply"
            write_ply(ply_path, sample_cloud())
            with self.assertRaises(SystemExit):
                render_main(
                    [
                        str(ply_path),
                        str(temporary_root / "out"),
                        "--elevation",
                        "90",
                    ]
                )
            leftovers = sorted(
                path.name
                for path in temporary_root.iterdir()
                if path.name != "cloud.ply"
            )

        self.assertEqual(leftovers, [])


class ArgumentValidationTests(unittest.TestCase):
    def test_report_rejects_unusable_arguments(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            volume_path = build_volume(Path(temporary_dir), frame_stride=2)
            cases = {
                "zero pixel step": ["--pixel-step", "0"],
                "negative pixel step": ["--pixel-step", "-4"],
                "negative offset": ["--held-out-offset", "-1"],
            }
            for name, extra in cases.items():
                with self.subTest(case=name):
                    # argparse prints its usage banner before exiting.
                    with (
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        report_main(
                            [str(FIXTURE), str(volume_path), *extra]
                        )

    def test_offset_outside_the_stride_is_refused(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            volume_path = build_volume(Path(temporary_dir), frame_stride=2)
            with self.assertRaises(SystemExit) as caught:
                report_main(
                    [
                        str(FIXTURE),
                        str(volume_path),
                        "--held-out-offset",
                        "3",
                    ]
                )
        self.assertIn(
            "outside the volume's frame_stride",
            str(caught.exception),
        )

    def test_renderer_rejects_unusable_arguments(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary_dir:
            temporary_root = Path(temporary_dir)
            ply_path = temporary_root / "cloud.ply"
            write_ply(ply_path, sample_cloud())
            cases = {
                "zero frames": ["--frames", "0"],
                "zero size": ["--size", "0"],
                "zero point size": ["--point-size", "0"],
                "fill above one": ["--fill", "1.5"],
                "fill of zero": ["--fill", "0"],
                "degenerate elevation": ["--elevation", "-90"],
            }
            for name, extra in cases.items():
                with self.subTest(case=name):
                    with (
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit),
                    ):
                        render_main(
                            [
                                str(ply_path),
                                str(temporary_root / f"out-{len(name)}"),
                                *extra,
                            ]
                        )


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
