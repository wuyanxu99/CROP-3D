"""Measured PPFD light-field fitting utilities.

The model treats the LED system as a measured black-box light source and fits
PPFD as a function of ``(setting, x, y, z)``.  The public query order matches the
point-cloud coordinate convention used in this project.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import types
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


@dataclass(frozen=True)
class LightFieldDomain:
    setting_min: float = 170.0
    setting_max: float = 240.0
    setting_center: float = 205.0
    setting_scale: float = 35.0
    x_min: float = 0.0
    x_max: float = 1.5
    y_min: float = 0.0
    y_max: float = 0.56
    z_min: float = 0.0
    z_max: float = 0.45


@dataclass(frozen=True)
class LightFieldConfig:
    domain: LightFieldDomain = LightFieldDomain()
    kernel: str = "thin_plate_spline"
    smoothing_candidates: tuple[float, ...] = (0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0)
    edge_decay: float = 0.75
    use_edge_prior: bool = True
    use_height_prior: bool = False
    height_prior_gain_fraction: float = 0.35
    height_prior_min_gain: float = 0.0
    setting_scale_mode: str = "setting_value"
    monotonic_violation_threshold: float = 0.05
    enforce_monotonic_z: bool | None = None
    height_knots: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45)


@dataclass(frozen=True)
class PPFDRecord:
    setting: float
    x: float
    y: float
    z: float
    ppfd: float
    source: str = "measured"


@dataclass
class FittedLightField:
    config: LightFieldConfig
    smoothing: float
    rbf: object
    training_summary: dict

    def predict_ppfd(self, setting, x, y, z, clip: bool = True):
        """Predict PPFD for scalar or array-like ``(setting, x, y, z)`` inputs."""
        np, _ = require_numeric_stack()
        arrays = np.broadcast_arrays(
            np.asarray(setting, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        flat = [a.reshape(-1) for a in arrays]
        setting_v, x_v, y_v, z_v = flat
        if clip:
            d = self.config.domain
            setting_v = np.clip(setting_v, d.setting_min, d.setting_max)
            x_v = np.clip(x_v, d.x_min, d.x_max)
            y_v = np.clip(y_v, d.y_min, d.y_max)
            z_v = np.clip(z_v, d.z_min, d.z_max)

        use_monotonic = bool(self.training_summary.get("enforce_monotonic_z", False))
        if use_monotonic:
            pred = self._predict_monotonic_z(setting_v, x_v, y_v, z_v)
        else:
            pred = self._predict_raw(setting_v, x_v, y_v, z_v)

        pred = np.maximum(pred, 0.0)
        return pred.reshape(arrays[0].shape)

    def _predict_raw(self, setting_v, x_v, y_v, z_v):
        np, _ = require_numeric_stack()
        points = normalize_points(setting_v, x_v, y_v, z_v, self.config.domain)
        return np.asarray(self.rbf(points), dtype=float)

    def _predict_monotonic_z(self, setting_v, x_v, y_v, z_v):
        np, _ = require_numeric_stack()
        knots = np.asarray(self.config.height_knots, dtype=float)
        out = np.empty_like(z_v, dtype=float)
        for i, (setting_i, x_i, y_i, z_i) in enumerate(zip(setting_v, x_v, y_v, z_v)):
            settings = np.full_like(knots, setting_i, dtype=float)
            xs = np.full_like(knots, x_i, dtype=float)
            ys = np.full_like(knots, y_i, dtype=float)
            raw = self._predict_raw(settings, xs, ys, knots)
            corrected = np.maximum.accumulate(raw)
            out[i] = np.interp(z_i, knots, corrected)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)


@dataclass
class LayeredLightField:
    config: LightFieldConfig
    smoothing_by_z: dict[float, float]
    rbf_by_z: dict[float, object]
    training_summary: dict

    def predict_ppfd(self, setting, x, y, z, clip: bool = True):
        """Predict PPFD with per-height RBF models and vertical interpolation."""
        np, _ = require_numeric_stack()
        arrays = np.broadcast_arrays(
            np.asarray(setting, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        flat = [a.reshape(-1) for a in arrays]
        setting_v, x_v, y_v, z_v = flat
        if clip:
            d = self.config.domain
            setting_v = np.clip(setting_v, d.setting_min, d.setting_max)
            x_v = np.clip(x_v, d.x_min, d.x_max)
            y_v = np.clip(y_v, d.y_min, d.y_max)
            z_v = np.clip(z_v, d.z_min, d.z_max)

        z_levels = np.asarray(sorted(self.rbf_by_z), dtype=float)
        layer_predictions = []
        for z_level in z_levels:
            layer_predictions.append(self._predict_layer(float(z_level), setting_v, x_v, y_v))
        stacked = np.vstack(layer_predictions)
        out = np.empty_like(z_v, dtype=float)
        for i, z_i in enumerate(z_v):
            out[i] = linear_interp_extrap(z_i, z_levels, stacked[:, i])
        out = np.maximum(out, 0.0)
        return out.reshape(arrays[0].shape)

    def _predict_layer(self, z_level: float, setting_v, x_v, y_v):
        np, _ = require_numeric_stack()
        d = self.config.domain
        points = np.column_stack(
            [
                (np.asarray(setting_v, dtype=float) - d.setting_center) / d.setting_scale,
                (np.asarray(x_v, dtype=float) - d.x_min) / (d.x_max - d.x_min),
                (np.asarray(y_v, dtype=float) - d.y_min) / (d.y_max - d.y_min),
            ]
        )
        return np.asarray(self.rbf_by_z[z_level](points), dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)


@dataclass
class ScaledSpatialLightField:
    config: LightFieldConfig
    smoothing: float
    rbf: object
    setting_scales: dict[float, float]
    training_summary: dict

    def predict_ppfd(self, setting, x, y, z, clip: bool = True):
        """Predict PPFD as setting-dependent scale times one shared spatial field."""
        np, _ = require_numeric_stack()
        arrays = np.broadcast_arrays(
            np.asarray(setting, dtype=float),
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        setting_v, x_v, y_v, z_v = [a.reshape(-1) for a in arrays]
        if clip:
            d = self.config.domain
            setting_v = np.clip(setting_v, d.setting_min, d.setting_max)
            x_v = np.clip(x_v, d.x_min, d.x_max)
            y_v = np.clip(y_v, d.y_min, d.y_max)
            z_v = np.clip(z_v, d.z_min, d.z_max)
        spatial = np.asarray(self.rbf(normalize_spatial_points(x_v, y_v, z_v, self.config.domain)), dtype=float)
        pred = self._setting_scale(setting_v) * spatial
        pred = np.maximum(pred, 0.0)
        return pred.reshape(arrays[0].shape)

    def _setting_scale(self, setting_v):
        np, _ = require_numeric_stack()
        known_settings = np.asarray(sorted(self.setting_scales), dtype=float)
        known_scales = np.asarray([self.setting_scales[float(s)] for s in known_settings], dtype=float)
        return np.interp(np.asarray(setting_v, dtype=float), known_settings, known_scales)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)


def load_light_field(path: str | Path) -> FittedLightField:
    legacy_pkg = sys.modules.setdefault("light", types.ModuleType("light"))
    setattr(legacy_pkg, "light_field_model", sys.modules[__name__])
    sys.modules.setdefault("light.light_field_model", sys.modules[__name__])
    with Path(path).open("rb") as f:
        return pickle.load(f)


def require_numeric_stack():
    try:
        import numpy as np
        from scipy.interpolate import RBFInterpolator
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This light-field fitter requires numpy and scipy. Install them with "
            "`pip install -e .` in the environment you "
            "use for fitting."
        ) from exc
    return np, RBFInterpolator


def read_ppfd_records(path: str | Path) -> list[PPFDRecord]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = _read_csv_rows(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm"}:
        rows = _read_xlsx_rows(path)
    else:
        raise ValueError(f"Unsupported PPFD table format: {path.suffix}")
    return standardize_ppfd_rows(rows)


def standardize_ppfd_rows(rows: Iterable[dict[str, object]]) -> list[PPFDRecord]:
    records: list[PPFDRecord] = []
    for row in rows:
        lowered = {str(k).strip().lower(): v for k, v in row.items()}
        if not lowered:
            continue
        try:
            setting = _as_float(lowered["setting"])
            x = _as_float(lowered.get("x", lowered.get("X")))
            y = _as_float(lowered.get("y", lowered.get("Y")))
            z = _as_float(lowered["z"])
            ppfd = _as_float(lowered.get("ppfd", lowered.get("value")))
        except (KeyError, TypeError, ValueError):
            continue
        if any(math.isnan(v) for v in (setting, x, y, z, ppfd)):
            continue
        records.append(PPFDRecord(setting=setting, x=x, y=y, z=z, ppfd=ppfd))
    return deduplicate_records(records)


def deduplicate_records(records: Iterable[PPFDRecord], ndigits: int = 9) -> list[PPFDRecord]:
    grouped: dict[tuple[float, float, float, float], list[PPFDRecord]] = defaultdict(list)
    for rec in records:
        key = (round(rec.setting, ndigits), round(rec.x, ndigits), round(rec.y, ndigits), round(rec.z, ndigits))
        grouped[key].append(rec)
    out = [
        PPFDRecord(
            setting=key[0],
            x=key[1],
            y=key[2],
            z=key[3],
            ppfd=sum(rec.ppfd for rec in vals) / len(vals),
            source=vals[0].source if all(rec.source == vals[0].source for rec in vals) else "mixed",
        )
        for key, vals in grouped.items()
    ]
    return sorted(out, key=lambda r: (r.setting, r.z, r.y, r.x))


def validate_records(records: Iterable[PPFDRecord], domain: LightFieldDomain) -> dict:
    records = list(records)
    if not records:
        raise ValueError("No valid PPFD records were found.")
    stats = {
        "count": len(records),
        "settings": sorted({r.setting for r in records}),
        "z_levels": sorted({r.z for r in records}),
        "plant_positions": len({(r.x, r.y) for r in records}),
        "ranges": {
            "setting": [min(r.setting for r in records), max(r.setting for r in records)],
            "x": [min(r.x for r in records), max(r.x for r in records)],
            "y": [min(r.y for r in records), max(r.y for r in records)],
            "z": [min(r.z for r in records), max(r.z for r in records)],
            "ppfd": [min(r.ppfd for r in records), max(r.ppfd for r in records)],
        },
    }
    warnings = []
    for name, low, high in (
        ("setting", domain.setting_min, domain.setting_max),
        ("x", domain.x_min, domain.x_max),
        ("y", domain.y_min, domain.y_max),
        ("z", domain.z_min, domain.z_max),
    ):
        vals = [getattr(r, name) for r in records]
        if min(vals) < low or max(vals) > high:
            warnings.append(f"{name} has measured values outside [{low}, {high}]")
    stats["warnings"] = warnings
    return stats


def fit_light_field(records: Iterable[PPFDRecord], config: LightFieldConfig | None = None) -> FittedLightField:
    require_numeric_stack()
    config = config or LightFieldConfig()
    real_records = list(records)
    validate_records(real_records, config.domain)
    cv = cross_validate_smoothing(real_records, config)
    smoothing = cv["selected_smoothing"]
    model_records = add_prior_records(real_records, config)
    rbf = _fit_rbf(model_records, config, smoothing)
    fitted = FittedLightField(
        config=config,
        smoothing=smoothing,
        rbf=rbf,
        training_summary={
            "record_count": len(real_records),
            "model_record_count": len(model_records),
            "pseudo_record_count": len(model_records) - len(real_records),
            "smoothing_cv": cv,
            "enforce_monotonic_z": False,
        },
    )
    prior_checks = check_priors(fitted, real_records, config)
    if config.enforce_monotonic_z is None:
        enforce_monotonic = prior_checks["height_non_decreasing"]["violation_ratio"] > config.monotonic_violation_threshold
    else:
        enforce_monotonic = bool(config.enforce_monotonic_z)
    fitted.training_summary["prior_checks_raw"] = prior_checks
    fitted.training_summary["enforce_monotonic_z"] = enforce_monotonic
    fitted.training_summary["prior_checks_final"] = check_priors(fitted, real_records, config)
    return fitted


def fit_layered_light_field(records: Iterable[PPFDRecord], config: LightFieldConfig | None = None) -> LayeredLightField:
    require_numeric_stack()
    config = config or LightFieldConfig()
    real_records = list(records)
    validate_records(real_records, config.domain)
    z_levels = sorted({r.z for r in real_records})
    cv = cross_validate_layered_smoothing(real_records, config)
    smoothing_by_z = cv["selected_smoothing_by_z"]
    rbf_by_z = {}
    model_record_count = 0
    pseudo_record_count = 0
    for z in z_levels:
        layer_records = [r for r in real_records if r.z == z]
        model_records = add_prior_records(layer_records, config, include_height_prior=False)
        model_record_count += len(model_records)
        pseudo_record_count += len(model_records) - len(layer_records)
        rbf_by_z[z] = _fit_rbf_xy(model_records, config, smoothing_by_z[z])
    fitted = LayeredLightField(
        config=config,
        smoothing_by_z=smoothing_by_z,
        rbf_by_z=rbf_by_z,
        training_summary={
            "model_type": "layered_xy_rbf_with_vertical_interpolation",
            "vertical_prediction": "linear interpolation inside measured z levels; linear extrapolation above highest measured z",
            "record_count": len(real_records),
            "model_record_count": model_record_count,
            "pseudo_record_count": pseudo_record_count,
            "smoothing_cv": cv,
            "enforce_monotonic_z": False,
        },
    )
    fitted.training_summary["prior_checks_final"] = check_priors(fitted, real_records, config)
    return fitted


def fit_scaled_spatial_light_field(
    records: Iterable[PPFDRecord],
    config: LightFieldConfig | None = None,
) -> ScaledSpatialLightField:
    require_numeric_stack()
    config = config or LightFieldConfig()
    real_records = list(records)
    validate_records(real_records, config.domain)
    cv = cross_validate_scaled_spatial_smoothing(real_records, config)
    smoothing = cv["selected_smoothing"]
    setting_scales = compute_setting_scales(real_records, config)
    model_records = add_prior_records(real_records, config)
    rbf = _fit_scaled_spatial_rbf(model_records, config, smoothing, setting_scales)
    fitted = ScaledSpatialLightField(
        config=config,
        smoothing=smoothing,
        rbf=rbf,
        setting_scales=setting_scales,
        training_summary={
            "model_type": "scaled_spatial_xyz_rbf",
            "prediction_form": "P(setting,x,y,z) = S(setting) * F(x,y,z)",
            "setting_scale_mode": config.setting_scale_mode,
            "setting_scales": setting_scales,
            "record_count": len(real_records),
            "model_record_count": len(model_records),
            "pseudo_record_count": len(model_records) - len(real_records),
            "smoothing_cv": cv,
            "enforce_monotonic_z": False,
        },
    )
    fitted.training_summary["prior_checks_final"] = check_priors(fitted, real_records, config)
    return fitted


def _fit_rbf(records: Iterable[PPFDRecord], config: LightFieldConfig, smoothing: float):
    np, RBFInterpolator = require_numeric_stack()
    records = list(records)
    points = normalize_records(records, config.domain)
    values = np.asarray([r.ppfd for r in records], dtype=float)
    return RBFInterpolator(points, values, kernel=config.kernel, smoothing=float(smoothing))


def _fit_scaled_spatial_rbf(
    records: Iterable[PPFDRecord],
    config: LightFieldConfig,
    smoothing: float,
    setting_scales: dict[float, float],
):
    np, RBFInterpolator = require_numeric_stack()
    records = list(records)
    points = normalize_records_spatial(records, config.domain)
    values = np.asarray([r.ppfd / _lookup_setting_scale(r.setting, setting_scales) for r in records], dtype=float)
    return RBFInterpolator(points, values, kernel=config.kernel, smoothing=float(smoothing))


def _fit_rbf_xy(records: Iterable[PPFDRecord], config: LightFieldConfig, smoothing: float):
    np, RBFInterpolator = require_numeric_stack()
    records = list(records)
    points = normalize_records_xy(records, config.domain)
    values = np.asarray([r.ppfd for r in records], dtype=float)
    return RBFInterpolator(points, values, kernel=config.kernel, smoothing=float(smoothing))


def normalize_records(records: Iterable[PPFDRecord], domain: LightFieldDomain):
    np, _ = require_numeric_stack()
    records = list(records)
    return normalize_points(
        np.asarray([r.setting for r in records], dtype=float),
        np.asarray([r.x for r in records], dtype=float),
        np.asarray([r.y for r in records], dtype=float),
        np.asarray([r.z for r in records], dtype=float),
        domain,
    )


def normalize_records_xy(records: Iterable[PPFDRecord], domain: LightFieldDomain):
    np, _ = require_numeric_stack()
    records = list(records)
    return np.column_stack(
        [
            (np.asarray([r.setting for r in records], dtype=float) - domain.setting_center) / domain.setting_scale,
            (np.asarray([r.x for r in records], dtype=float) - domain.x_min) / (domain.x_max - domain.x_min),
            (np.asarray([r.y for r in records], dtype=float) - domain.y_min) / (domain.y_max - domain.y_min),
        ]
    )


def normalize_records_spatial(records: Iterable[PPFDRecord], domain: LightFieldDomain):
    np, _ = require_numeric_stack()
    records = list(records)
    return normalize_spatial_points(
        np.asarray([r.x for r in records], dtype=float),
        np.asarray([r.y for r in records], dtype=float),
        np.asarray([r.z for r in records], dtype=float),
        domain,
    )


def normalize_spatial_points(x, y, z, domain: LightFieldDomain):
    np, _ = require_numeric_stack()
    return np.column_stack(
        [
            (np.asarray(x, dtype=float) - domain.x_min) / (domain.x_max - domain.x_min),
            (np.asarray(y, dtype=float) - domain.y_min) / (domain.y_max - domain.y_min),
            (np.asarray(z, dtype=float) - domain.z_min) / (domain.z_max - domain.z_min),
        ]
    )


def normalize_points(setting, x, y, z, domain: LightFieldDomain):
    np, _ = require_numeric_stack()
    return np.column_stack(
        [
            (np.asarray(setting, dtype=float) - domain.setting_center) / domain.setting_scale,
            (np.asarray(x, dtype=float) - domain.x_min) / (domain.x_max - domain.x_min),
            (np.asarray(y, dtype=float) - domain.y_min) / (domain.y_max - domain.y_min),
            (np.asarray(z, dtype=float) - domain.z_min) / (domain.z_max - domain.z_min),
        ]
    )


def add_edge_prior_records(records: Iterable[PPFDRecord], config: LightFieldConfig) -> list[PPFDRecord]:
    """Add low-confidence boundary pseudo-observations for edge attenuation."""
    real = list(records)
    grouped: dict[tuple[float, float], list[PPFDRecord]] = defaultdict(list)
    for rec in real:
        grouped[(rec.setting, rec.z)].append(rec)

    d = config.domain
    decay = config.edge_decay
    pseudo: list[PPFDRecord] = []
    for (setting, z), group in grouped.items():
        by_y: dict[float, list[PPFDRecord]] = defaultdict(list)
        for rec in group:
            by_y[rec.y].append(rec)
        for y, row in by_y.items():
            left = min(row, key=lambda r: r.x)
            right = max(row, key=lambda r: r.x)
            pseudo.append(PPFDRecord(setting, d.x_min, y, z, max(0.0, left.ppfd * decay), "edge_prior"))
            pseudo.append(PPFDRecord(setting, d.x_max, y, z, max(0.0, right.ppfd * decay), "edge_prior"))

        y_min = min(by_y)
        y_max = max(by_y)
        bottom = sorted(by_y[y_min], key=lambda r: r.x)
        top = sorted(by_y[y_max], key=lambda r: r.x)
        for rec in _representative_edge_points(bottom):
            pseudo.append(PPFDRecord(setting, rec.x, d.y_min, z, max(0.0, rec.ppfd * decay), "edge_prior"))
        for rec in _representative_edge_points(top):
            pseudo.append(PPFDRecord(setting, rec.x, d.y_max, z, max(0.0, rec.ppfd * decay), "edge_prior"))

    return real + deduplicate_records(pseudo)


def add_height_prior_records(records: Iterable[PPFDRecord], config: LightFieldConfig) -> list[PPFDRecord]:
    """Add low-confidence pseudo-observations for non-decreasing vertical PPFD."""
    real = list(records)
    if not real:
        return []

    z_knots = sorted(float(z) for z in config.height_knots)
    slopes = _height_prior_slopes(real, config)
    grouped: dict[tuple[float, float, float], list[PPFDRecord]] = defaultdict(list)
    for rec in real:
        if rec.source != "measured":
            continue
        grouped[(rec.setting, rec.x, rec.y)].append(rec)

    pseudo: list[PPFDRecord] = []
    for (setting, x, y), profile in grouped.items():
        profile = sorted(profile, key=lambda r: r.z)
        existing_z = {round(rec.z, 9) for rec in profile}
        for z in z_knots:
            if round(z, 9) in existing_z:
                continue
            lower = [rec for rec in profile if rec.z < z]
            higher = [rec for rec in profile if rec.z > z]
            if not lower:
                continue
            lower_rec = max(lower, key=lambda r: r.z)
            dz = z - lower_rec.z
            gain = max(config.height_prior_min_gain, slopes.get(setting, slopes["global"]) * dz)
            ppfd = lower_rec.ppfd + gain
            if higher:
                higher_rec = min(higher, key=lambda r: r.z)
                # Keep the pseudo point compatible with the next measured point.
                ppfd = min(ppfd, higher_rec.ppfd)
            pseudo.append(PPFDRecord(setting, x, y, z, max(0.0, ppfd), "height_prior"))
    return deduplicate_records(pseudo)


def add_prior_records(
    records: Iterable[PPFDRecord],
    config: LightFieldConfig,
    include_height_prior: bool = True,
) -> list[PPFDRecord]:
    out = list(records)
    if include_height_prior and config.use_height_prior:
        out = out + add_height_prior_records(out, config)
    if config.use_edge_prior:
        out = add_edge_prior_records(out, config)
    return out


def _height_prior_slopes(records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict[float | str, float]:
    grouped: dict[tuple[float, float, float], list[PPFDRecord]] = defaultdict(list)
    for rec in records:
        if rec.source == "measured":
            grouped[(rec.setting, rec.x, rec.y)].append(rec)

    slopes_by_setting: dict[float, list[float]] = defaultdict(list)
    all_slopes: list[float] = []
    for (setting, _, _), profile in grouped.items():
        profile = sorted(profile, key=lambda r: r.z)
        for a, b in zip(profile, profile[1:]):
            dz = b.z - a.z
            if dz <= 0:
                continue
            slope = (b.ppfd - a.ppfd) / dz
            if slope > 0:
                slopes_by_setting[setting].append(slope)
                all_slopes.append(slope)

    global_slope = _median(all_slopes) * config.height_prior_gain_fraction if all_slopes else 0.0
    slopes: dict[float | str, float] = {"global": global_slope}
    for setting, vals in slopes_by_setting.items():
        slopes[setting] = _median(vals) * config.height_prior_gain_fraction if vals else global_slope
    return slopes


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _representative_edge_points(row: list[PPFDRecord], max_points: int = 5) -> list[PPFDRecord]:
    if len(row) <= max_points:
        return row
    if max_points <= 1:
        return [row[len(row) // 2]]
    idx = [round(i * (len(row) - 1) / (max_points - 1)) for i in range(max_points)]
    return [row[i] for i in sorted(set(idx))]


def cross_validate_smoothing(records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict:
    require_numeric_stack()
    records = list(records)
    settings = sorted({r.setting for r in records})
    positions = sorted({(r.x, r.y) for r in records})
    results = []
    for smoothing in config.smoothing_candidates:
        y_true = []
        y_pred = []
        failures = 0
        for pos in positions:
            train = [r for r in records if (r.x, r.y) != pos]
            test = [r for r in records if (r.x, r.y) == pos]
            try:
                train_with_prior = add_prior_records(train, config)
                rbf = _fit_rbf(train_with_prior, config, smoothing)
                fold_model = FittedLightField(
                    config=config,
                    smoothing=smoothing,
                    rbf=rbf,
                    training_summary={"enforce_monotonic_z": False},
                )
                preds = fold_model.predict_ppfd(
                    [r.setting for r in test],
                    [r.x for r in test],
                    [r.y for r in test],
                    [r.z for r in test],
                )
                y_true.extend(r.ppfd for r in test)
                y_pred.extend(float(v) for v in preds)
            except Exception:
                failures += 1
        metrics = regression_metrics(y_true, y_pred)
        metrics["smoothing"] = smoothing
        metrics["fold_failures"] = failures
        results.append(metrics)

    valid = [r for r in results if not math.isnan(r["rmse"])]
    if not valid:
        raise RuntimeError("All smoothing cross-validation candidates failed.")
    selected = min(valid, key=lambda r: r["rmse"])
    return {"selected_smoothing": selected["smoothing"], "selected_metrics": selected, "candidates": results, "settings": settings}


def cross_validate_layered_smoothing(records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict:
    require_numeric_stack()
    records = list(records)
    settings = sorted({r.setting for r in records})
    z_levels = sorted({r.z for r in records})
    selected_by_z = {}
    candidates_by_z = {}
    all_true = []
    all_pred = []
    for z in z_levels:
        layer = [r for r in records if r.z == z]
        positions = sorted({(r.x, r.y) for r in layer})
        results = []
        for smoothing in config.smoothing_candidates:
            y_true = []
            y_pred = []
            failures = 0
            for pos in positions:
                train = [r for r in layer if (r.x, r.y) != pos]
                test = [r for r in layer if (r.x, r.y) == pos]
                try:
                    train_with_prior = add_edge_prior_records(train, config) if config.use_edge_prior else train
                    rbf = _fit_rbf_xy(train_with_prior, config, smoothing)
                    tmp = LayeredLightField(
                        config=config,
                        smoothing_by_z={z: smoothing},
                        rbf_by_z={z: rbf},
                        training_summary={"enforce_monotonic_z": False},
                    )
                    preds = tmp.predict_ppfd(
                        [r.setting for r in test],
                        [r.x for r in test],
                        [r.y for r in test],
                        [r.z for r in test],
                    )
                    y_true.extend(r.ppfd for r in test)
                    y_pred.extend(float(v) for v in preds)
                except Exception:
                    failures += 1
            metrics = regression_metrics(y_true, y_pred)
            metrics["smoothing"] = smoothing
            metrics["fold_failures"] = failures
            results.append(metrics)
        valid = [r for r in results if not math.isnan(r["rmse"])]
        if not valid:
            raise RuntimeError(f"All layered smoothing candidates failed for z={z}.")
        selected = min(valid, key=lambda r: r["rmse"])
        selected_by_z[z] = selected["smoothing"]
        candidates_by_z[z] = {"selected_metrics": selected, "candidates": results}

        # Recompute out-of-position predictions for the selected smoothing.
        for pos in positions:
            train = [r for r in layer if (r.x, r.y) != pos]
            test = [r for r in layer if (r.x, r.y) == pos]
            train_with_prior = add_prior_records(train, config, include_height_prior=False)
            rbf = _fit_rbf_xy(train_with_prior, config, selected["smoothing"])
            tmp = LayeredLightField(
                config=config,
                smoothing_by_z={z: selected["smoothing"]},
                rbf_by_z={z: rbf},
                training_summary={"enforce_monotonic_z": False},
            )
            preds = tmp.predict_ppfd(
                [r.setting for r in test],
                [r.x for r in test],
                [r.y for r in test],
                [r.z for r in test],
            )
            all_true.extend(r.ppfd for r in test)
            all_pred.extend(float(v) for v in preds)
    return {
        "selected_smoothing_by_z": selected_by_z,
        "selected_metrics": regression_metrics(all_true, all_pred),
        "candidates_by_z": candidates_by_z,
        "settings": settings,
        "z_levels": z_levels,
    }


def cross_validate_scaled_spatial_smoothing(records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict:
    require_numeric_stack()
    records = list(records)
    settings = sorted({r.setting for r in records})
    positions = sorted({(r.x, r.y) for r in records})
    results = []
    for smoothing in config.smoothing_candidates:
        y_true = []
        y_pred = []
        failures = 0
        for pos in positions:
            train = [r for r in records if (r.x, r.y) != pos]
            test = [r for r in records if (r.x, r.y) == pos]
            try:
                setting_scales = compute_setting_scales(train, config)
                train_with_prior = add_prior_records(train, config)
                rbf = _fit_scaled_spatial_rbf(train_with_prior, config, smoothing, setting_scales)
                fold_model = ScaledSpatialLightField(
                    config=config,
                    smoothing=smoothing,
                    rbf=rbf,
                    setting_scales=setting_scales,
                    training_summary={"enforce_monotonic_z": False},
                )
                preds = fold_model.predict_ppfd(
                    [r.setting for r in test],
                    [r.x for r in test],
                    [r.y for r in test],
                    [r.z for r in test],
                )
                y_true.extend(r.ppfd for r in test)
                y_pred.extend(float(v) for v in preds)
            except Exception:
                failures += 1
        metrics = regression_metrics(y_true, y_pred)
        metrics["smoothing"] = smoothing
        metrics["fold_failures"] = failures
        results.append(metrics)

    valid = [r for r in results if not math.isnan(r["rmse"])]
    if not valid:
        raise RuntimeError("All scaled-spatial smoothing cross-validation candidates failed.")
    selected = min(valid, key=lambda r: r["rmse"])
    return {"selected_smoothing": selected["smoothing"], "selected_metrics": selected, "candidates": results, "settings": settings}


def compute_setting_scales(records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict[float, float]:
    records = [r for r in records if r.source == "measured"]
    settings = sorted({float(r.setting) for r in records})
    if not settings:
        raise ValueError("Cannot compute setting scales without measured records.")
    mode = config.setting_scale_mode
    if mode == "setting_value":
        return {setting: setting / config.domain.setting_center for setting in settings}
    if mode == "train_mean":
        means = {}
        for setting in settings:
            vals = [r.ppfd for r in records if r.setting == setting]
            means[setting] = sum(vals) / len(vals)
        center = means.get(config.domain.setting_center)
        if center is None:
            center = sum(means.values()) / len(means)
        return {setting: means[setting] / center for setting in settings}
    raise ValueError(f"Unsupported setting_scale_mode: {mode}")


def _lookup_setting_scale(setting: float, setting_scales: dict[float, float]) -> float:
    keys = sorted(setting_scales)
    if setting in setting_scales:
        return setting_scales[setting]
    if len(keys) == 1:
        return setting_scales[keys[0]]
    if setting <= keys[0]:
        left, right = keys[0], keys[1]
    elif setting >= keys[-1]:
        left, right = keys[-2], keys[-1]
    else:
        left, right = keys[0], keys[-1]
        for idx in range(1, len(keys)):
            if setting <= keys[idx]:
                left, right = keys[idx - 1], keys[idx]
                break
    if right == left:
        return setting_scales[left]
    t = (setting - left) / (right - left)
    return setting_scales[left] + t * (setting_scales[right] - setting_scales[left])


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    y_true = [float(v) for v in y_true]
    y_pred = [float(v) for v in y_pred]
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan, "pearson_r": math.nan}
    errors = [p - t for t, p in zip(y_true, y_pred)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean_true = sum(y_true) / n
    sst = sum((t - mean_true) ** 2 for t in y_true)
    sse = sum(e * e for e in errors)
    r2 = 1.0 - sse / sst if sst > 0 else math.nan
    mean_pred = sum(y_pred) / n
    cov = sum((t - mean_true) * (p - mean_pred) for t, p in zip(y_true, y_pred))
    var_true = sum((t - mean_true) ** 2 for t in y_true)
    var_pred = sum((p - mean_pred) ** 2 for p in y_pred)
    pearson = cov / math.sqrt(var_true * var_pred) if var_true > 0 and var_pred > 0 else math.nan
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson}


def linear_interp_extrap(x: float, xp, fp) -> float:
    """One-dimensional linear interpolation with linear extrapolation at both ends."""
    xp = [float(v) for v in xp]
    fp = [float(v) for v in fp]
    if len(xp) == 1:
        return fp[0]
    if x <= xp[0]:
        x0, x1 = xp[0], xp[1]
        y0, y1 = fp[0], fp[1]
    elif x >= xp[-1]:
        x0, x1 = xp[-2], xp[-1]
        y0, y1 = fp[-2], fp[-1]
    else:
        for idx in range(1, len(xp)):
            if x <= xp[idx]:
                x0, x1 = xp[idx - 1], xp[idx]
                y0, y1 = fp[idx - 1], fp[idx]
                break
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def check_priors(model: FittedLightField, records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict:
    np, _ = require_numeric_stack()
    records = list(records)
    settings = sorted({r.setting for r in records})
    xs = np.linspace(config.domain.x_min, config.domain.x_max, 31)
    ys = np.linspace(config.domain.y_min, config.domain.y_max, 15)
    zs = np.asarray(config.height_knots, dtype=float)
    height_total = 0
    height_ok = 0
    for setting in settings:
        for x in xs:
            for y in ys:
                pred = model.predict_ppfd(setting, x, y, zs, clip=True)
                diffs = np.diff(pred)
                height_total += len(diffs)
                height_ok += int(np.sum(diffs >= -1e-6))
    edge = check_edge_trend(model, records, config)
    ok_ratio = height_ok / height_total if height_total else math.nan
    return {
        "height_non_decreasing": {
            "ok_ratio": ok_ratio,
            "violation_ratio": 1.0 - ok_ratio if not math.isnan(ok_ratio) else math.nan,
            "checked_pairs": height_total,
        },
        "edge_trend": edge,
    }


def check_edge_trend(model: FittedLightField, records: Iterable[PPFDRecord], config: LightFieldConfig) -> dict:
    np, _ = require_numeric_stack()
    settings = sorted({r.setting for r in records})
    xs = np.linspace(config.domain.x_min, config.domain.x_max, 41)
    ys = np.linspace(config.domain.y_min, config.domain.y_max, 17)
    zs = np.asarray(config.height_knots, dtype=float)
    edge_values = []
    center_values = []
    edge_margin = 0.08
    for setting in settings:
        for z in zs:
            for x in xs:
                for y in ys:
                    val = float(model.predict_ppfd(setting, x, y, z))
                    near_edge = (
                        x <= config.domain.x_min + edge_margin
                        or x >= config.domain.x_max - edge_margin
                        or y <= config.domain.y_min + edge_margin
                        or y >= config.domain.y_max - edge_margin
                    )
                    if near_edge:
                        edge_values.append(val)
                    elif 0.35 <= x <= 1.15 and 0.14 <= y <= 0.42:
                        center_values.append(val)
    edge_mean = sum(edge_values) / len(edge_values) if edge_values else math.nan
    center_mean = sum(center_values) / len(center_values) if center_values else math.nan
    return {
        "edge_mean": edge_mean,
        "center_mean": center_mean,
        "edge_lower_than_center": edge_mean < center_mean if not math.isnan(edge_mean + center_mean) else None,
    }


def write_records_csv(records: Iterable[PPFDRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "x", "y", "z", "ppfd"])
        writer.writeheader()
        for rec in records:
            writer.writerow({"setting": rec.setting, "x": rec.x, "y": rec.y, "z": rec.z, "ppfd": rec.ppfd})


def write_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def config_to_dict(config: LightFieldConfig) -> dict:
    data = asdict(config)
    data["smoothing_candidates"] = list(config.smoothing_candidates)
    data["height_knots"] = list(config.height_knots)
    return data


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_xlsx_rows(path: Path) -> list[dict[str, object]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf, ns)
        sheet_name = _first_sheet_name(zf)
        root = ElementTree.fromstring(zf.read(sheet_name))
    rows = []
    for row in root.findall(".//a:sheetData/a:row", ns):
        values = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            values[col] = _cell_value(cell, shared_strings, ns)
        rows.append(values)
    if not rows:
        return []
    cols = sorted({col for row in rows for col in row}, key=lambda c: (len(c), c))
    headers = [str(rows[0].get(col, "")).strip() for col in cols]
    out = []
    for row in rows[1:]:
        if not any(str(row.get(col, "")).strip() for col in cols):
            continue
        out.append({headers[i]: row.get(cols[i], "") for i in range(len(cols)) if headers[i]})
    return out


def _read_shared_strings(zf: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("a:si", ns):
        strings.append("".join((t.text or "") for t in item.findall(".//a:t", ns)))
    return strings


def _first_sheet_name(zf: zipfile.ZipFile) -> str:
    candidates = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
    if not candidates:
        raise ValueError("No worksheet XML found in xlsx file.")
    return candidates[0]


def _cell_value(cell, shared_strings: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", ns)
    if cell_type == "inlineStr":
        return "".join((t.text or "") for t in cell.findall(".//a:t", ns))
    if value is None:
        return ""
    text = value.text or ""
    if cell_type == "s":
        return shared_strings[int(text)]
    return text


def _as_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    return float(text)
