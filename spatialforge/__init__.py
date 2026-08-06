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
from .tsdf_block_traversal import (
    TsdfBlockTraversalReceipt,
    traverse_tsdf_block_voxels_from_context,
)
from .tsdf_plan_traversal import (
    MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES,
    TsdfPlanTraversalReceipt,
    traverse_tsdf_plan_blocks_from_context,
)
from .tsdf_observation_block_rays import (
    MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES,
    TsdfObservationBlockRayReceipt,
    TsdfObservationBlockRayStatus,
    TsdfObservationBlockRayTraceReceipt,
    trace_tsdf_observation_block_rays_from_context,
)
from .tsdf_plan_block_ray_survey import (
    MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES,
    TsdfPlanBlockRaySurveyReceipt,
    survey_tsdf_plan_block_rays_from_context,
)
from .tsdf_pixel_footprint_coverage import (
    MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS,
    TsdfPixelFootprintCoverageReceipt,
    TsdfPixelFootprintStatus,
    evaluate_tsdf_pixel_footprint_coverage_from_context,
)
from .tsdf_observation_footprint import (
    MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS,
    TsdfObservationFootprintReceipt,
    survey_tsdf_observation_pixel_footprints_from_context,
)
from .tsdf_plan_footprint_survey import (
    MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS,
    TsdfPlanFootprintSurveyReceipt,
    survey_tsdf_plan_pixel_footprints_from_context,
)
from .tsdf_replay_depth_context import (
    MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES,
    TsdfReplayDepthContext,
    TsdfReplayDepthObservation,
    TsdfReplayDepthStatus,
    build_tsdf_replay_depth_context,
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
    evaluate_tsdf_voxel_contribution_from_context,
)
from .tsdf_block_cross_view import (
    MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES,
    TsdfBlockCrossViewReceipt,
    classify_tsdf_block_voxels_across_observations_from_context,
)
from .tsdf_domain_cross_view import (
    MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES,
    TsdfCoverageDomainCrossViewReceipt,
    sweep_tsdf_coverage_domain_cross_view_from_context,
)
from .tsdf_expanded_plan import (
    TsdfExpandedPlanReport,
    write_tsdf_expanded_block_plan,
)
from .tsdf_plan_expansion import (
    TsdfPlanExpansionProposal,
    propose_tsdf_plan_expansion_from_domain,
)
from .tsdf_voxel_cross_view import (
    MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS,
    TsdfVoxelCrossViewReceipt,
    TsdfVoxelCrossViewVerdict,
    classify_tsdf_voxel_across_observations_from_context,
)
from .tsdf_voxel_sampling import (
    TsdfVoxelSamplingReceipt,
    TsdfVoxelSamplingStatus,
    classify_tsdf_voxel_sampling_from_context,
)
from .tsdf_voxel_traversal import (
    TsdfVoxelTraversalReceipt,
    traverse_tsdf_voxel_observations,
    traverse_tsdf_voxel_observations_from_context,
)
from .tsdf_voxel_update import (
    MAX_TSDF_VOXEL_WEIGHT,
    TsdfVoxelUpdateReceipt,
    apply_tsdf_voxel_contribution,
    apply_tsdf_voxel_contribution_from_context,
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
    "TsdfBlockCrossViewReceipt",
    "TsdfBlockPlan",
    "TsdfBlockPlanReport",
    "TsdfBlockStorage",
    "TsdfBlockTraversalReceipt",
    "TsdfContributionStatus",
    "TsdfExpandedPlanReport",
    "TsdfCoverageDomainCrossViewReceipt",
    "TsdfObservationBlockRayReceipt",
    "TsdfObservationBlockRayStatus",
    "TsdfObservationBlockRayTraceReceipt",
    "TsdfObservationFootprintReceipt",
    "TsdfPixelFootprintCoverageReceipt",
    "TsdfPixelFootprintStatus",
    "TsdfPlanBlockRaySurveyReceipt",
    "TsdfPlanExpansionProposal",
    "TsdfPlanFootprintSurveyReceipt",
    "TsdfPlanTraversalReceipt",
    "TsdfReplayDepthContext",
    "TsdfReplayDepthObservation",
    "TsdfReplayDepthStatus",
    "TsdfVoxelCrossViewReceipt",
    "TsdfVoxelCrossViewVerdict",
    "TsdfVoxelSamplingReceipt",
    "TsdfVoxelSamplingStatus",
    "TsdfVoxelTraversalReceipt",
    "TsdfVoxelUpdateReceipt",
    "TsdfVoxelAddress",
    "TsdfVoxelContribution",
    "TumImportReport",
    "allocate_empty_tsdf_blocks",
    "apply_tsdf_voxel_contribution",
    "apply_tsdf_voxel_contribution_from_context",
    "build_tsdf_replay_depth_context",
    "classify_tsdf_block_voxels_across_observations_from_context",
    "classify_tsdf_voxel_across_observations_from_context",
    "classify_tsdf_voxel_sampling_from_context",
    "compose_tsdf_global_voxel_index",
    "evaluate_tsdf_pixel_footprint_coverage_from_context",
    "evaluate_tsdf_voxel_contribution",
    "evaluate_tsdf_voxel_contribution_from_context",
    "extract_surface_points",
    "extract_triangle_mesh",
    "import_tum_dataset",
    "integrate_sparse_tsdf",
    "integrate_tsdf",
    "infer_tsdf_bounds",
    "load_scan_session",
    "load_tsdf_block_plan",
    "locate_tsdf_voxel",
    "MAX_TSDF_BLOCK_CROSS_VIEW_OUTCOMES",
    "MAX_TSDF_COVERAGE_DOMAIN_CROSS_VIEW_OUTCOMES",
    "MAX_TSDF_VOXEL_CROSS_VIEW_OBSERVATIONS",
    "MAX_TSDF_VOXEL_WEIGHT",
    "MAX_TSDF_REPLAY_DEPTH_CONTEXT_BYTES",
    "MAX_TSDF_OBSERVATION_BLOCK_RAY_OUTCOMES",
    "MAX_TSDF_OBSERVATION_FOOTPRINT_CANDIDATE_BLOCKS",
    "MAX_TSDF_PIXEL_FOOTPRINT_CANDIDATE_BLOCKS",
    "MAX_TSDF_PLAN_BLOCK_RAY_SURVEY_OUTCOMES",
    "MAX_TSDF_PLAN_FOOTPRINT_CANDIDATE_BLOCKS",
    "MAX_TSDF_PLAN_TRAVERSAL_OUTCOMES",
    "plan_tsdf_blocks",
    "propose_tsdf_plan_expansion_from_domain",
    "reconstruct_point_cloud",
    "replay_session",
    "survey_tsdf_observation_pixel_footprints_from_context",
    "survey_tsdf_plan_block_rays_from_context",
    "survey_tsdf_plan_pixel_footprints_from_context",
    "sweep_tsdf_coverage_domain_cross_view_from_context",
    "trace_tsdf_observation_block_rays_from_context",
    "traverse_tsdf_block_voxels_from_context",
    "traverse_tsdf_plan_blocks_from_context",
    "traverse_tsdf_voxel_observations",
    "traverse_tsdf_voxel_observations_from_context",
    "verify_tsdf_block_plan_replay",
    "write_tsdf_expanded_block_plan",
]

__version__ = "0.1.0"
