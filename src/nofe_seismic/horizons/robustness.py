"""Noise stress, overlapping tile-seam and 3D held-out slab checks for horizons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from nofe import NOFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from nofe_seismic.data.attributes import extract_seismic_attributes
from nofe_seismic.data.volumes import LazySeismicVolume, SeismicVolume, volume_to_points
from nofe_seismic.graph import (
    SeismicPointGraph,
    SharedStandardizer,
    coord_map_2d,
    embed_numpy,
)
from nofe_seismic.metrics import norm_rmse
from nofe_seismic.types import IndexRange, RowDict, Stride3D

from .boundaries import (
    evaluate_boundary_map,
    evaluate_horizon_curve,
    extract_boundary_ridges,
    extract_true_interfaces,
    extract_unit_top_depths,
    multiscale_boundary_contrast,
)
from .config import HorizonConfig
from .dp import coupled_monotonic_boundaries_2d
from .evals import NativeSlice, SliceEmbedModels, seed_for_unit_top
from .rgt import (
    estimate_structure_tensor_2d,
    extract_horizon_iso_rgt,
    extract_surface_iso_rgt_3d,
    rgt_aligned_smooth,
    solve_rgt_2d,
    solve_rgt_3d,
)

_LARGE_SLAB_THRESHOLD = 400
_NOISE_HALO = 5
_MIN_RADIUS_OVERLAP_MARGIN = 2
_OVERLAP_LO_FRAC = 0.35
_OVERLAP_HI_FRAC = 0.65
_NMS_SIZE = 11
_TOP_K_RIDGES = 3
_SLAB_SMOOTH_SIGMA_U = 1.5
_SLAB_SMOOTH_SIGMA_Z = 1.0
_PROB_EPS = 1e-6


@dataclass(frozen=True)
class TileSeamResult:
    """Overlapping two-tile seam consistency metrics and contrast mosaics."""

    mid_x: float
    shared_interior_points: int
    nofe_seam_norm_rmse: float
    pca_tile_seam_norm_rmse: float
    contrast_mosaic: dict[str, np.ndarray]


@dataclass(frozen=True)
class Slab3DPlotData:
    """Arrays and title needed to plot the 3D held-out slab surface check."""

    i_coords: np.ndarray
    x_coords: np.ndarray
    true_surf: np.ndarray
    pred_surf: np.ndarray
    err_title: str


@dataclass(frozen=True)
class Slab3DEvalResult:
    """3D held-out slab summary metrics and optional surface plot inputs."""

    metrics: dict[str, object]
    plot_data: Slab3DPlotData | None = None


@dataclass(frozen=True)
class _NoisySectionGrids:
    amp_noisy: np.ndarray
    x_raw: np.ndarray
    z_nofe: np.ndarray
    z_pca: np.ndarray


def _noisy_section_grids(
    volume: SeismicVolume | LazySeismicVolume,
    models: SliceEmbedModels,
    *,
    inline_idx: int,
    sl_clean: NativeSlice,
    cfg: HorizonConfig,
) -> _NoisySectionGrids:
    ni, _, nz = volume.shape
    lat = cfg.slice_lateral_stride
    rng = np.random.default_rng(cfg.seed)
    i0 = max(0, inline_idx - _NOISE_HALO)
    i1 = min(ni, inline_idx + _NOISE_HALO + 1)
    center_offset = inline_idx - i0

    slab_clean = np.asarray(volume.amplitude[i0:i1, ::lat, :], dtype=np.float32)
    amp_scale = (
        volume.amp_scale
        if isinstance(volume, LazySeismicVolume)
        else (float(volume.amplitude.std()) or 1.0)
    )
    slab_noisy = slab_clean + rng.normal(
        scale=cfg.noise_sigma * amp_scale, size=slab_clean.shape
    ).astype(np.float32)
    feats_noisy = extract_seismic_attributes(slab_noisy, scale=amp_scale).features
    xs_noisy = models.scaler.transform_features(
        feats_noisy[center_offset].reshape(-1, feats_noisy.shape[-1])
    )
    ps_clean = models.scaler.transform_locations(sl_clean.locations)
    g_noisy = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xs_noisy, ps_clean)
    nu = int(sl_clean.u_coords.size)
    return _NoisySectionGrids(
        amp_noisy=slab_noisy[center_offset],
        x_raw=xs_noisy.reshape(nu, nz, -1),
        z_nofe=embed_numpy(models.nofe, g_noisy).reshape(nu, nz, -1),
        z_pca=models.pca.transform(xs_noisy).astype(np.float32).reshape(nu, nz, -1),
    )


def evaluate_noise_stress(
    volume: SeismicVolume | LazySeismicVolume,
    models: SliceEmbedModels,
    *,
    inline_idx: int,
    sl_clean: NativeSlice,
    cfg: HorizonConfig,
    unit_ids: Sequence[int],
) -> list[RowDict]:
    """Inject 3D Gaussian trace noise and evaluate RGT and boundary degradation."""
    noisy = _noisy_section_grids(
        volume, models, inline_idx=inline_idx, sl_clean=sl_clean, cfg=cfg
    )
    lbl = sl_clean.labels
    z_coords = sl_clean.z_coords
    true_iface = extract_true_interfaces(lbl) if lbl is not None else None
    true_tops = (
        extract_unit_top_depths(lbl, unit_ids, z_coords) if lbl is not None else {}
    )

    stress_rows: list[RowDict] = []
    for m_name, grid in (
        ("NOFE-8D", noisy.z_nofe),
        ("PCA-8D", noisy.z_pca),
        ("Raw-11D", noisy.x_raw),
        ("Amplitude-1D", noisy.amp_noisy[:, :, None]),
    ):
        contrast = multiscale_boundary_contrast(grid, windows=cfg.contrast_windows)
        ridges = extract_boundary_ridges(
            contrast, nms_size=_NMS_SIZE, top_k=_TOP_K_RIDGES
        )
        b_ev = (
            evaluate_boundary_map(true_iface, contrast, ridges, z_coords=z_coords)
            if true_iface is not None
            else {}
        )
        dip_f = estimate_structure_tensor_2d(grid, max_slope=cfg.max_slope)
        rgt = solve_rgt_2d(dip_f, w_dip=cfg.rgt_w_dip, w_mono=cfg.rgt_w_mono)
        if lbl is None:
            continue
        for uid in unit_ids:
            seed = seed_for_unit_top(lbl, uid)
            if seed is None:
                continue
            pred_c = extract_horizon_iso_rgt(rgt, seed.u, seed.z, z_coords=z_coords)
            h_ev = evaluate_horizon_curve(true_tops[uid], pred_c)
            stress_rows.append(
                {
                    "test": f"noise_sigma_{cfg.noise_sigma}",
                    "method": m_name,
                    "horizon": f"Top_Unit_{uid}",
                    "mae": h_ev["mae"],
                    "medae": h_ev["medae"],
                    "within_3px": h_ev["within_3px"],
                    "boundary_roc_auc": b_ev.get("boundary_roc_auc", float("nan")),
                }
            )
    return stress_rows


def _build_contrast_mosaic(
    u_full: np.ndarray,
    nz: int,
    mid_x: int,
    tile_embeds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    windows: tuple[int, ...],
) -> dict[str, np.ndarray]:
    contrast_mosaic: dict[str, np.ndarray] = {}
    for m_name, (zw_flat, ze_flat, locs_w, locs_e) in tile_embeds.items():
        uw = np.unique(locs_w[:, 1]).size
        ue = np.unique(locs_e[:, 1]).size
        cw = multiscale_boundary_contrast(zw_flat.reshape(uw, nz, -1), windows=windows)
        ce = multiscale_boundary_contrast(ze_flat.reshape(ue, nz, -1), windows=windows)
        xw_u = np.unique(locs_w[:, 1])
        xe_u = np.unique(locs_e[:, 1])
        mos = np.zeros((u_full.size, nz - 1), dtype=np.float32)
        for idx_u, x_val in enumerate(u_full):
            if x_val < mid_x:
                mos[idx_u] = cw[int(np.argmin(np.abs(xw_u - x_val)))]
            else:
                mos[idx_u] = ce[int(np.argmin(np.abs(xe_u - x_val)))]
        contrast_mosaic[m_name] = mos
    return contrast_mosaic


def evaluate_tile_seam(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    *,
    inline_idx: int,
    sl_clean: NativeSlice,
    cfg: HorizonConfig,
) -> TileSeamResult:
    """Compare West/East overlapping tile seam consistency for NOFE vs PCA."""
    _, nx, nz = volume.shape
    lat = cfg.slice_lateral_stride
    mid_x = nx // 2
    ov_lo = max(0, (int(_OVERLAP_LO_FRAC * nx) // lat) * lat)
    ov_hi = min(nx, max(ov_lo + lat * 2, int(_OVERLAP_HI_FRAC * nx)))

    pts_w = volume_to_points(
        volume,
        inline_range=IndexRange(inline_idx, inline_idx + 1),
        crossline_range=IndexRange(0, ov_hi),
        stride=Stride3D(1, lat, 1),
    )
    pts_e = volume_to_points(
        volume,
        inline_range=IndexRange(inline_idx, inline_idx + 1),
        crossline_range=IndexRange(ov_lo, nx),
        stride=Stride3D(1, lat, 1),
    )
    xw = scaler.transform_features(pts_w.features)
    pw = scaler.transform_locations(pts_w.locations)
    xe = scaler.transform_features(pts_e.features)
    pe = scaler.transform_locations(pts_e.locations)

    gw = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xw, pw)
    ge = SeismicPointGraph(
        mode="radius", radius=cfg.radius_slice, normalize=False
    ).build(xe, pe)
    zn_w = embed_numpy(nofe8, gw)
    zn_e = embed_numpy(nofe8, ge)
    zp_w = (
        PCA(n_components=cfg.out_dim, random_state=cfg.seed)
        .fit_transform(xw)
        .astype(np.float32)
    )
    zp_e = (
        PCA(n_components=cfg.out_dim, random_state=cfg.seed)
        .fit_transform(xe)
        .astype(np.float32)
    )

    mw = coord_map_2d(pts_w.locations[:, [1, 2]])
    me = coord_map_2d(pts_e.locations[:, [1, 2]])
    margin = min(max(_MIN_RADIUS_OVERLAP_MARGIN, lat * 2), max(1, (ov_hi - ov_lo) // 4))
    shared = [k for k in mw if k in me and (ov_lo + margin) <= k[0] <= (ov_hi - margin)]
    iw = [mw[k] for k in shared]
    ie = [me[k] for k in shared]
    nofe_seam = norm_rmse(zn_w[iw], zn_e[ie]) if shared else float("nan")
    pca_seam = norm_rmse(zp_w[iw], zp_e[ie]) if shared else float("nan")

    contrast_mosaic = _build_contrast_mosaic(
        sl_clean.u_coords,
        nz,
        mid_x,
        {
            "NOFE-8D": (zn_w, zn_e, pts_w.locations, pts_e.locations),
            "PCA-8D (Per-Tile)": (zp_w, zp_e, pts_w.locations, pts_e.locations),
        },
        cfg.contrast_windows,
    )
    return TileSeamResult(
        mid_x=float(mid_x),
        shared_interior_points=len(shared),
        nofe_seam_norm_rmse=round(float(nofe_seam), 4),
        pca_tile_seam_norm_rmse=round(float(pca_seam), 4),
        contrast_mosaic=contrast_mosaic,
    )


def _eval_slab_probe_dp(
    clf: Pipeline,
    z_slab_nofe: np.ndarray,
    rgt_3d: np.ndarray,
    z_vals: np.ndarray,
    unit_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    s_ni, s_nx, s_nz, _ = z_slab_nofe.shape
    probs_3d = clf.predict_proba(
        z_slab_nofe.reshape(-1, z_slab_nofe.shape[-1])
    ).reshape(s_ni, s_nx, s_nz, -1)
    logp_3d = np.log(np.clip(probs_3d, _PROB_EPS, 1.0)).astype(np.float32)
    classes = np.asarray(clf.classes_, dtype=np.int64)
    seg_3d = np.zeros((s_ni, s_nx, s_nz), dtype=np.int64)
    for idx_i in range(s_ni):
        logp_sl = rgt_aligned_smooth(
            logp_3d[idx_i],
            rgt_3d[idx_i],
            sigma_u=_SLAB_SMOOTH_SIGMA_U,
            sigma_z=_SLAB_SMOOTH_SIGMA_Z,
        )
        _, seg_3d[idx_i] = coupled_monotonic_boundaries_2d(
            logp_sl,
            classes,
            rgt_3d[idx_i],
            z_coords=z_vals,
            lam_lat=4.0,
        )
    return extract_unit_top_depths(seg_3d, unit_ids, z_vals)


def evaluate_3d_slab(
    volume: SeismicVolume | LazySeismicVolume,
    scaler: SharedStandardizer,
    nofe8: NOFE,
    *,
    split_i: int,
    unit_ids: Sequence[int],
    probes_cache: dict[str, Pipeline] | None,
    cfg: HorizonConfig,
) -> Slab3DEvalResult:
    """Reconstruct and evaluate 3D horizon surfaces on the held-out 3D slab."""
    ni = volume.shape.ni
    slab_stride = (
        Stride3D(8, 16, 2) if ni >= _LARGE_SLAB_THRESHOLD else Stride3D(2, 2, 1)
    )
    slab_pts = volume_to_points(
        volume, inline_range=IndexRange(split_i, ni), stride=slab_stride
    )
    i_vals = np.unique(slab_pts.locations[:, 0])
    x_vals = np.unique(slab_pts.locations[:, 1])
    z_vals = np.unique(slab_pts.locations[:, 2])
    s_ni, s_nx, s_nz = i_vals.size, x_vals.size, z_vals.size

    xs_slab = scaler.transform_features(slab_pts.features)
    ps_slab = scaler.transform_locations(slab_pts.locations)
    g_slab = SeismicPointGraph(
        mode="radius", radius=cfg.radius_train, normalize=False
    ).build(xs_slab, ps_slab)
    z_slab_nofe = embed_numpy(nofe8, g_slab).reshape(s_ni, s_nx, s_nz, -1)
    rgt_3d = solve_rgt_3d(
        z_slab_nofe, max_slope=cfg.max_slope, w_dip=cfg.rgt_w_dip, w_mono=cfg.rgt_w_mono
    )

    metrics: dict[str, object] = {
        "slab_shape": [s_ni, s_nx, s_nz],
        "stride": list(slab_stride),
    }
    if slab_pts.labels is None or not unit_ids:
        return Slab3DEvalResult(metrics=metrics)

    lbl_3d = slab_pts.labels.reshape(s_ni, s_nx, s_nz)
    uid_main = unit_ids[0]
    true_surf = extract_unit_top_depths(lbl_3d, unit_ids, z_vals)[uid_main]
    valid_ix = np.argwhere(np.isfinite(true_surf))
    if valid_ix.size == 0:
        return Slab3DEvalResult(metrics=metrics)

    si_mid, sx_mid = valid_ix[len(valid_ix) // 2]
    seed_z_idx = float(
        np.interp(
            float(true_surf[si_mid, sx_mid]),
            z_vals,
            np.arange(s_nz, dtype=np.float32),
        )
    )
    pred_surf_rgt = extract_surface_iso_rgt_3d(
        rgt_3d, int(si_mid), int(sx_mid), seed_z_idx, z_coords=z_vals
    )
    rgt_3d_ev = evaluate_horizon_curve(true_surf, pred_surf_rgt)
    metrics[f"rgt_3d_Top_Unit_{uid_main}"] = rgt_3d_ev

    pred_surf = pred_surf_rgt
    err_title = f"3D RGT MAE={rgt_3d_ev['mae']:.2f}, MedAE={rgt_3d_ev['medae']:.2f}"
    if probes_cache is not None and "NOFE-8D" in probes_cache:
        pred_tops_3d = _eval_slab_probe_dp(
            probes_cache["NOFE-8D"], z_slab_nofe, rgt_3d, z_vals, unit_ids
        )
        dp_3d_ev = evaluate_horizon_curve(true_surf, pred_tops_3d[uid_main])
        metrics[f"probe_dp_3d_Top_Unit_{uid_main}"] = dp_3d_ev
        pred_surf = pred_tops_3d[uid_main]
        err_title = (
            f"RGT MAE={rgt_3d_ev['mae']:.2f} | Probe-DP MAE={dp_3d_ev['mae']:.2f}"
        )

    return Slab3DEvalResult(
        metrics=metrics,
        plot_data=Slab3DPlotData(
            i_coords=i_vals,
            x_coords=x_vals,
            true_surf=true_surf,
            pred_surf=pred_surf,
            err_title=err_title,
        ),
    )
