"""SpatialForge's deterministic offline mapping foundations."""

from .model import Observation, ScanSession
from .mesh import TriangleMeshReport, extract_triangle_mesh
from .point_cloud import PointCloudReport, reconstruct_point_cloud
from .replay import ReplayResult, replay_session
from .session_loader import load_scan_session
from .sparse_tsdf import integrate_sparse_tsdf
from .surface import SurfacePointReport, extract_surface_points
from .tsdf import TsdfReport, integrate_tsdf
from .tsdf_block_plan import TsdfBlockPlanReport, plan_tsdf_blocks
from .tsdf_block_plan_loader import (
    TsdfBlockPlan,
    load_tsdf_block_plan,
    verify_tsdf_block_plan_replay,
)
from .tsdf_bounds import TsdfBoundsReport, infer_tsdf_bounds
from .tum_importer import TumImportReport, import_tum_dataset

__all__ = [
    "Observation",
    "PointCloudReport",
    "ReplayResult",
    "ScanSession",
    "SurfacePointReport",
    "TriangleMeshReport",
    "TsdfReport",
    "TsdfBoundsReport",
    "TsdfBlockPlan",
    "TsdfBlockPlanReport",
    "TumImportReport",
    "extract_surface_points",
    "extract_triangle_mesh",
    "import_tum_dataset",
    "integrate_sparse_tsdf",
    "integrate_tsdf",
    "infer_tsdf_bounds",
    "load_scan_session",
    "load_tsdf_block_plan",
    "plan_tsdf_blocks",
    "reconstruct_point_cloud",
    "replay_session",
    "verify_tsdf_block_plan_replay",
]

__version__ = "0.1.0"
