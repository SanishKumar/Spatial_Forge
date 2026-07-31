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
from .tsdf_block_storage import (
    TsdfBlockStorage,
    allocate_empty_tsdf_blocks,
)
from .tsdf_voxel_address import (
    TsdfVoxelAddress,
    compose_tsdf_global_voxel_index,
    locate_tsdf_voxel,
)
from .tsdf_voxel_contribution import (
    TsdfContributionStatus,
    TsdfVoxelContribution,
    evaluate_tsdf_voxel_contribution,
)
from .tsdf_voxel_traversal import (
    TsdfVoxelTraversalReceipt,
    traverse_tsdf_voxel_observations,
)
from .tsdf_voxel_update import (
    MAX_TSDF_VOXEL_WEIGHT,
    TsdfVoxelUpdateReceipt,
    apply_tsdf_voxel_contribution,
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
    "TsdfBlockStorage",
    "TsdfContributionStatus",
    "TsdfVoxelTraversalReceipt",
    "TsdfVoxelUpdateReceipt",
    "TsdfVoxelAddress",
    "TsdfVoxelContribution",
    "TumImportReport",
    "allocate_empty_tsdf_blocks",
    "apply_tsdf_voxel_contribution",
    "compose_tsdf_global_voxel_index",
    "evaluate_tsdf_voxel_contribution",
    "extract_surface_points",
    "extract_triangle_mesh",
    "import_tum_dataset",
    "integrate_sparse_tsdf",
    "integrate_tsdf",
    "infer_tsdf_bounds",
    "load_scan_session",
    "load_tsdf_block_plan",
    "locate_tsdf_voxel",
    "MAX_TSDF_VOXEL_WEIGHT",
    "plan_tsdf_blocks",
    "reconstruct_point_cloud",
    "replay_session",
    "traverse_tsdf_voxel_observations",
    "verify_tsdf_block_plan_replay",
]

__version__ = "0.1.0"
