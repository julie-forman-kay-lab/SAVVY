"""
Surface Accessibility Via Vector Yield (SAVVY), v0.4.0.
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import math
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "MG": 1.73,
    "ZN": 1.39,
    "CA": 1.94,
    "FE": 1.56,
}

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "MSE": "M", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

BACKBONE_HEAVY_ATOMS = {"N", "CA", "C", "O", "OXT"}

CONDITION_ORDER = ["APO", "BOUND_ONLY", "BOUND_FULL"]
CONDITION_LABEL = {
    "APO": "APO",
    "BOUND_ONLY": "Bound-state target only",
    "BOUND_FULL": "Bound complex (full)",
}


@dataclass
class AtomTable:
    coords: np.ndarray
    radii: np.ndarray
    chains: np.ndarray
    resnums: np.ndarray
    icodes: np.ndarray
    resnames: np.ndarray
    atom_names: np.ndarray
    elements: np.ndarray

    def subset(self, mask: np.ndarray) -> "AtomTable":
        return AtomTable(
            coords=self.coords[mask],
            radii=self.radii[mask],
            chains=self.chains[mask],
            resnums=self.resnums[mask],
            icodes=self.icodes[mask],
            resnames=self.resnames[mask],
            atom_names=self.atom_names[mask],
            elements=self.elements[mask],
        )

    @property
    def n_atoms(self) -> int:
        return int(self.coords.shape[0])


@dataclass
class SiteGeometry:
    origin: np.ndarray
    frame: np.ndarray
    resname: str
    residue_one_letter: str
    target_definition: str
    target_atom_name: str
    target_atom_names: str
    target_atom_count: int
    frame_definition: str


def parse_csv_ints(text: str) -> List[int]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return vals


def parse_csv_floats(text: str) -> List[float]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one number.")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Surface Accessibility Via Vector Yield (SAVVY) analysis "
            "for target residues in apo and partner-bound ensembles."
        ),
    )
    p.add_argument(
        "--redo-plots",
        metavar="PREVIOUS_OUTPUT_DIR",
        default=None,
        help=(
            "Plot-only mode: read a completed SAVVY output directory and regenerate "
            "figures in --outdir without repeating accessibility calculations."
        ),
    )
    p.add_argument("--apo-dir", default=None)
    p.add_argument("--bound-dir", default=None)
    p.add_argument("--apo-glob", default="*.pdb")
    p.add_argument("--bound-glob", default="*.pdb")
    p.add_argument("--apo-target-chain", default="A")
    p.add_argument("--bound-target-chain", default="B")
    p.add_argument(
        "--bound-partner-chains",
        default="all",
        help="'all' means every non-target chain; otherwise comma-separated chain IDs.",
    )
    p.add_argument(
        "--target-residues", "--phosphosites",
        dest="target_residues",
        type=parse_csv_ints,
        default=[38, 47, 66, 71, 84],
        help=(
            "Comma-separated target residue numbers to analyze."
        ),
    )
    p.add_argument(
        "--region-name",
        default=None,
        help=(
            "Optional label for a residue group, for example FEMDI. When supplied with "
            "--region-residues, conformer-level region mean/minimum endpoints are written."
        ),
    )
    p.add_argument(
        "--region-residues",
        type=parse_csv_ints,
        default=None,
        help="Optional comma-separated target residues forming one binding region.",
    )
    p.add_argument(
        "--target-mode",
        choices=["auto", "sidechain-centroid", "st-oxygen"],
        default="auto",
        help=(
            "auto uses OG/OG1 for SER/THR and the geometric side-chain heavy-atom "
            "centroid otherwise; sidechain-centroid forces centroids for all residues; "
            "st-oxygen requires every target to be SER/THR."
        ),
    )
    p.add_argument(
        "--gly-target-mode",
        choices=["error", "ca"],
        default="error",
        help=(
            "GLY has no side-chain heavy-atom centroid. 'error' fails that conformer; "
            "'ca' explicitly uses Cα as a fallback."
        ),
    )
    p.add_argument("--probe-radii", type=parse_csv_floats, default=[2, 4, 6, 8, 10])
    p.add_argument("--approach-lengths", type=parse_csv_floats, default=[10, 20, 30])
    p.add_argument(
        "--exclude-sequence-windows",
        type=parse_csv_ints,
        default=[1, 2, 3],
        help="Sensitivity grid of target local sequence half-windows excluded from calculation.",
    )
    p.add_argument("--primary-probe-radius", type=float, default=4.0)
    p.add_argument("--primary-approach-length", type=float, default=20.0)
    p.add_argument("--primary-exclude-window", type=int, default=2)
    p.add_argument("--n-directions", type=int, default=2048)
    p.add_argument(
        "--direction-convergence-counts",
        type=parse_csv_ints,
        default=[256, 512, 1024, 2048],
        help="Direction counts evaluated at the primary parameter setting.",
    )
    p.add_argument(
        "--ray-start",
        type=float,
        default=2.5,
        help="Distance from the target origin at which the swept probe begins.",
    )
    p.add_argument(
        "--minimum-meaningful-delta",
        type=float,
        default=0.01,
        help="Absolute accessible-fraction loss used to define a meaningfully masked conformer.",
    )
    p.add_argument(
        "--apo-weights-csv",
        default=None,
        help="Optional CSV with columns filename and weight for APO conformers.",
    )
    p.add_argument(
        "--bound-weights-csv",
        default=None,
        help="Optional CSV with columns filename and weight for bound conformers.",
    )
    p.add_argument(
        "--include-hetero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include HETATM records as obstacles; protein ATOM records are always included.",
    )
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--figure-dpi", type=int, default=300)
    p.add_argument(
        "--residue-label-shift", "--plot-residue-label-shift",
        dest="residue_label_shift",
        type=int,
        default=0,
        help=(
            "Integer added only to residue numbers displayed in plots. The target residue "
            "identity and all calculated data remain unchanged; e.g. Thr38 with -1 is T37."
        ),
    )
    p.add_argument("--outdir", required=True)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument(
        "--save-directional-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args()

    if args.figure_dpi <= 0:
        p.error("--figure-dpi must be > 0.")

    if args.redo_plots is not None:
        source = Path(args.redo_plots).expanduser()
        destination = Path(args.outdir).expanduser()
        if not source.is_dir():
            p.error(f"--redo-plots directory does not exist: {source}")
        if source.resolve() == destination.resolve():
            p.error("--outdir must be a new directory, not the same directory as --redo-plots.")
        return args

    if not args.apo_dir or not args.bound_dir:
        p.error("--apo-dir and --bound-dir are required unless --redo-plots is used.")

    args.target_residues = sorted(set(int(x) for x in args.target_residues))
    if args.region_residues is not None:
        args.region_residues = sorted(set(int(x) for x in args.region_residues))
    args.probe_radii = sorted(set(float(x) for x in args.probe_radii))
    args.approach_lengths = sorted(set(float(x) for x in args.approach_lengths))
    args.exclude_sequence_windows = sorted(set(int(x) for x in args.exclude_sequence_windows))
    args.direction_convergence_counts = sorted(set(int(x) for x in args.direction_convergence_counts))

    if (args.region_name is None) != (args.region_residues is None):
        p.error("--region-name and --region-residues must be supplied together.")
    if args.region_residues is not None:
        missing_region = sorted(set(args.region_residues) - set(args.target_residues))
        if missing_region:
            p.error(
                "Every --region-residues entry must also be included in --target-residues; "
                f"missing: {missing_region}"
            )
        if len(args.region_residues) < 2:
            p.error("--region-residues should contain at least two residues.")

    if args.n_directions < 128:
        p.error("--n-directions should be at least 128.")
    if args.ray_start < 0:
        p.error("--ray-start must be >= 0.")
    if any(x <= args.ray_start for x in args.approach_lengths):
        p.error("Every --approach-length must be greater than --ray-start.")
    if any(x <= 0 for x in args.probe_radii):
        p.error("Every --probe-radius must be > 0.")
    if any(x < 0 for x in args.exclude_sequence_windows):
        p.error("Every exclusion window must be >= 0.")
    if args.minimum_meaningful_delta < 0:
        p.error("--minimum-meaningful-delta must be >= 0.")
    if max(args.direction_convergence_counts) != args.n_directions:
        args.direction_convergence_counts.append(args.n_directions)
        args.direction_convergence_counts = sorted(set(args.direction_convergence_counts))
    if any(x < 64 or x > args.n_directions for x in args.direction_convergence_counts):
        p.error("Direction convergence counts must be between 64 and --n-directions.")

    def ensure_present(value: float, allowed: Sequence[float], name: str) -> None:
        if not any(abs(float(value) - float(x)) < 1e-8 for x in allowed):
            p.error(f"{name}={value} must be one of {list(allowed)}")

    ensure_present(args.primary_probe_radius, args.probe_radii, "--primary-probe-radius")
    ensure_present(args.primary_approach_length, args.approach_lengths, "--primary-approach-length")
    if args.primary_exclude_window not in args.exclude_sequence_windows:
        p.error("--primary-exclude-window must be included in --exclude-sequence-windows")
    return args


def infer_element(atom_name: str, element_field: str) -> str:
    element = str(element_field).strip().upper()
    if element:
        return element
    name = "".join(ch for ch in str(atom_name).strip() if ch.isalpha()).upper()
    if not name:
        return "C"
    if len(name) >= 2 and name[:2] in VDW_RADII:
        return name[:2]
    return name[0]


def parse_pdb_heavy_atoms(path: str, include_hetero: bool = False) -> AtomTable:
    coords: List[Tuple[float, float, float]] = []
    radii: List[float] = []
    chains: List[str] = []
    resnums: List[int] = []
    icodes: List[str] = []
    resnames: List[str] = []
    atom_names: List[str] = []
    elements: List[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "HETATM" and not include_hetero:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            altloc = line[16:17].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            element = infer_element(atom_name, line[76:78] if len(line) >= 78 else "")
            if element == "H" or atom_name.upper().startswith("H"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resnum = int(line[22:26])
            except ValueError:
                continue

            chain = line[21:22].strip()
            icode = line[26:27].strip()
            resname = line[17:20].strip().upper()
            coords.append((x, y, z))
            radii.append(float(VDW_RADII.get(element, 1.70)))
            chains.append(chain)
            resnums.append(resnum)
            icodes.append(icode)
            resnames.append(resname)
            atom_names.append(atom_name)
            elements.append(element)

    if not coords:
        raise RuntimeError(f"No heavy protein coordinates parsed from {path}")

    return AtomTable(
        coords=np.asarray(coords, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        chains=np.asarray(chains, dtype=object),
        resnums=np.asarray(resnums, dtype=np.int32),
        icodes=np.asarray(icodes, dtype=object),
        resnames=np.asarray(resnames, dtype=object),
        atom_names=np.asarray(atom_names, dtype=object),
        elements=np.asarray(elements, dtype=object),
    )


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * i / golden_ratio
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.column_stack([x, y, z])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs.astype(np.float64)


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise RuntimeError(f"Cannot define local target-residue frame: degenerate {label} vector")
    return vector / norm


def _residue_atom_coord(atoms: AtomTable, mask: np.ndarray, atom_name: str) -> Optional[np.ndarray]:
    idx = np.flatnonzero(mask & (atoms.atom_names == atom_name))
    if idx.size == 0:
        return None
    return atoms.coords[idx[0]].copy()


def find_target_geometry(
    atoms: AtomTable,
    chain: str,
    site: int,
    target_mode: str,
    gly_target_mode: str,
    ) -> SiteGeometry:
        mask = (atoms.chains == chain) & (atoms.resnums == int(site))
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            raise RuntimeError(f"Residue {chain}:{site} not found")

        resname = str(atoms.resnames[idxs[0]]).upper()
        one = AA3_TO_1.get(resname, "X")
        mode = str(target_mode)
        if mode == "auto":
            resolved_mode = "st-oxygen" if one in {"S", "T"} else "sidechain-centroid"
        else:
            resolved_mode = mode

        ca = _residue_atom_coord(atoms, mask, "CA")
        cb = _residue_atom_coord(atoms, mask, "CB")
        n_atom = _residue_atom_coord(atoms, mask, "N")
        c_atom = _residue_atom_coord(atoms, mask, "C")
        if ca is None:
            raise RuntimeError(f"CA atom required to define target geometry for {chain}:{site}")

        if resolved_mode == "st-oxygen":
            if one not in {"S", "T"}:
                raise RuntimeError(
                    f"--target-mode st-oxygen requires SER/THR, but {chain}:{site} is {resname}"
                )
            preferred = ["OG1"] if one == "T" else ["OG"]
            preferred += ["OG", "OG1"]
            oxygen_name: Optional[str] = None
            origin: Optional[np.ndarray] = None
            for atom_name in preferred:
                origin = _residue_atom_coord(atoms, mask, atom_name)
                if origin is not None:
                    oxygen_name = atom_name
                    break
            if origin is None or oxygen_name is None:
                raise RuntimeError(
                    f"No side-chain O-gamma atom found for {chain}:{site} ({resname}); "
                    "expected OG for SER or OG1 for THR"
                )
            if cb is None:
                raise RuntimeError(f"CB atom required for SER/THR oxygen frame at {chain}:{site}")

            z_axis = _normalize(origin - cb, "CB-to-Ogamma")
            x_hint = cb - ca
            x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
            if np.linalg.norm(x_proj) < 1e-8 and n_atom is not None:
                x_hint = ca - n_atom
                x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
            x_axis = _normalize(x_proj, "projected CA-to-CB")
            y_axis = _normalize(np.cross(z_axis, x_axis), "local y")
            x_axis = _normalize(np.cross(y_axis, z_axis), "re-orthogonalized local x")
            frame = np.column_stack([x_axis, y_axis, z_axis])
            return SiteGeometry(
                origin=origin,
                frame=frame,
                resname=resname,
                residue_one_letter=one,
                target_definition="ser_thr_sidechain_oxygen",
                target_atom_name=oxygen_name,
                target_atom_names=oxygen_name,
                target_atom_count=1,
                frame_definition="z=CB_to_Ogamma;x=projected_CA_to_CB;y=z_cross_x",
            )

        if resolved_mode != "sidechain-centroid":
            raise RuntimeError(f"Unsupported resolved target mode: {resolved_mode}")

        sidechain_mask = mask & (~np.isin(atoms.atom_names, np.asarray(sorted(BACKBONE_HEAVY_ATOMS), dtype=object)))
        side_idx = np.flatnonzero(sidechain_mask)
        if side_idx.size == 0:
            if one == "G" and gly_target_mode == "ca":
                if n_atom is None or c_atom is None:
                    raise RuntimeError(
                        f"N, CA, and C atoms are required for GLY C-alpha fallback at {chain}:{site}"
                    )
                origin = ca.copy()
                z_axis = _normalize(ca - n_atom, "N-to-CA GLY fallback")
                x_hint = c_atom - ca
                x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
                x_axis = _normalize(x_proj, "projected CA-to-C GLY fallback")
                y_axis = _normalize(np.cross(z_axis, x_axis), "local y")
                x_axis = _normalize(np.cross(y_axis, z_axis), "re-orthogonalized local x")
                frame = np.column_stack([x_axis, y_axis, z_axis])
                return SiteGeometry(
                    origin=origin,
                    frame=frame,
                    resname=resname,
                    residue_one_letter=one,
                    target_definition="gly_ca_fallback",
                    target_atom_name="CA",
                    target_atom_names="CA",
                    target_atom_count=1,
                    frame_definition="z=N_to_CA;x=projected_CA_to_C;y=z_cross_x",
                )
            if one == "G":
                raise RuntimeError(
                    f"GLY {chain}:{site} has no side-chain heavy-atom centroid; "
                    "use --gly-target-mode ca only if a Cα fallback is acceptable"
                )
            raise RuntimeError(
                f"No side-chain heavy atoms found for {chain}:{site} ({resname}); "
                "the structure may have incomplete side-chain coordinates"
            )

        side_coords = atoms.coords[side_idx]
        origin = np.mean(side_coords, axis=0)
        z_axis = _normalize(origin - ca, "CA-to-sidechain-centroid")

        x_hint: Optional[np.ndarray] = None
        x_label = ""
        if c_atom is not None:
            x_hint = c_atom - ca
            x_label = "CA-to-carbonyl-C"
        elif n_atom is not None:
            x_hint = ca - n_atom
            x_label = "N-to-CA"
        elif cb is not None:
            x_hint = cb - ca
            x_label = "CA-to-CB"
        if x_hint is None:
            raise RuntimeError(f"Cannot define local frame for {chain}:{site}; missing backbone reference atoms")

        x_proj = x_hint - np.dot(x_hint, z_axis) * z_axis
        if np.linalg.norm(x_proj) < 1e-8:
            fallbacks: List[Tuple[str, Optional[np.ndarray]]] = [
                ("N-to-CA", None if n_atom is None else ca - n_atom),
                ("CA-to-CB", None if cb is None else cb - ca),
            ]
            for label, candidate in fallbacks:
                if candidate is None:
                    continue
                candidate_proj = candidate - np.dot(candidate, z_axis) * z_axis
                if np.linalg.norm(candidate_proj) >= 1e-8:
                    x_proj = candidate_proj
                    x_label = label
                    break
        x_axis = _normalize(x_proj, f"projected {x_label}")
        y_axis = _normalize(np.cross(z_axis, x_axis), "local y")
        x_axis = _normalize(np.cross(y_axis, z_axis), "re-orthogonalized local x")
        frame = np.column_stack([x_axis, y_axis, z_axis])

        atom_names = sorted(str(x) for x in atoms.atom_names[side_idx])
        return SiteGeometry(
            origin=origin,
            frame=frame,
            resname=resname,
            residue_one_letter=one,
            target_definition="sidechain_heavy_atom_geometric_centroid",
            target_atom_name="SC_CENTROID",
            target_atom_names=";".join(atom_names),
            target_atom_count=int(side_idx.size),
            frame_definition=f"z=CA_to_sidechain_centroid;x=projected_{x_label.replace('-', '_')};y=z_cross_x",
        )


def parse_partner_chains(text: str, atoms: AtomTable, prot_chain: str) -> List[str]:
    if str(text).strip().lower() == "all":
        return sorted({str(c) for c in atoms.chains if str(c) != str(prot_chain)})
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def obstacle_mask(
    atoms: AtomTable,
    included_chains: Sequence[str],
    prot_chain: str,
    site: int,
    exclude_window: int,
    ) -> np.ndarray:
        included = np.isin(atoms.chains, np.asarray(list(included_chains), dtype=object))
        local_target = (
            (atoms.chains == prot_chain)
            & (atoms.resnums >= int(site) - int(exclude_window))
            & (atoms.resnums <= int(site) + int(exclude_window))
        )
        return included & (~local_target)


def largest_open_cone_half_angle(
    directions: np.ndarray,
    accessible: np.ndarray,
    ) -> float:
        n_accessible = int(np.count_nonzero(accessible))
        if n_accessible == 0:
            return 0.0
        if n_accessible == len(accessible):
            return 180.0
        blocked_dirs = directions[~accessible]
        accessible_dirs = directions[accessible]
        tree = cKDTree(blocked_dirs)
        chord, _ = tree.query(accessible_dirs, k=1)
        chord = np.clip(chord, 0.0, 2.0)
        angle = 2.0 * np.arcsin(chord / 2.0)
        return float(np.degrees(np.max(angle)))


def analyze_condition_site(
    site_coord: np.ndarray,
    obstacles: AtomTable,
    global_directions: np.ndarray,
    local_directions: np.ndarray,
    probe_radii: Sequence[float],
    approach_lengths: Sequence[float],
    ray_start: float,
    prot_chain: str,
    primary_probe_radius: float,
    primary_approach_length: float,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        directions = global_directions
        if obstacles.n_atoms == 0:
            rows = []
            for length in approach_lengths:
                for probe in probe_radii:
                    rows.append({
                        "probe_radius_A": float(probe),
                        "approach_length_A": float(length),
                        "n_obstacle_atoms": 0,
                        "accessible_fraction": 1.0,
                        "blocked_fraction": 0.0,
                        "mean_clear_distance_A": float(length),
                        "median_clear_distance_A": float(length),
                        "largest_open_cone_half_angle_deg": 180.0,
                        "largest_open_cone_solid_angle_fraction": 1.0,
                        "target_first_block_fraction": 0.0,
                        "partner_first_block_fraction": 0.0,
                        "partner_fraction_among_blocked": 0.0,
                    })
            return rows, np.ones(len(directions), dtype=bool)

        max_probe = float(max(probe_radii))
        max_length = float(max(approach_lengths))
        rel_all = obstacles.coords - site_coord[np.newaxis, :]
        center_dist = np.linalg.norm(rel_all, axis=1)
        candidate = center_dist <= (max_length + max_probe + obstacles.radii + 1e-8)
        obs = obstacles.subset(candidate)
        rel = obs.coords - site_coord[np.newaxis, :]
        r2 = np.sum(rel * rel, axis=1)

        proj = directions @ rel.T
        perp2 = np.maximum(0.0, r2[np.newaxis, :] - proj * proj)
        rows: List[Dict[str, Any]] = []
        selected_map: Optional[np.ndarray] = None

        for length in approach_lengths:
            length = float(length)
            t_clamped = np.clip(proj, float(ray_start), length)
            segment_d2 = np.maximum(
                r2[np.newaxis, :] + t_clamped * t_clamped - 2.0 * proj * t_clamped,
                0.0,
            )
            for probe in probe_radii:
                probe = float(probe)
                radius2 = (obs.radii + probe) ** 2
                collision = segment_d2 < radius2[np.newaxis, :]
                blocked = np.any(collision, axis=1)
                accessible = ~blocked

                root_term = np.maximum(0.0, radius2[np.newaxis, :] - perp2)
                intersects_infinite = perp2 < radius2[np.newaxis, :]
                root = np.sqrt(root_term)
                entry = proj - root
                exit_ = proj + root
                hit = intersects_infinite & (exit_ >= float(ray_start)) & (entry <= length)
                first = np.where(hit, np.maximum(entry, float(ray_start)), np.inf)
                first_idx = np.argmin(first, axis=1)
                first_dist = first[np.arange(len(directions)), first_idx]
                no_hit = ~np.isfinite(first_dist)
                first_dist[no_hit] = length
                first_dist = np.clip(first_dist, float(ray_start), length)

                first_chain = np.full(len(directions), "", dtype=object)
                has_hit = ~no_hit
                if np.any(has_hit):
                    first_chain[has_hit] = obs.chains[first_idx[has_hit]]
                target_first = has_hit & (first_chain == prot_chain)
                partner_first = has_hit & (first_chain != prot_chain)
                blocked_count = int(np.count_nonzero(has_hit))
                partner_among_blocked = (
                    float(np.count_nonzero(partner_first) / blocked_count)
                    if blocked_count else 0.0
                )

                cone_angle = largest_open_cone_half_angle(local_directions, accessible)
                cone_fraction = (
                    1.0 if cone_angle >= 180.0
                    else float((1.0 - math.cos(math.radians(cone_angle))) / 2.0)
                )
                rows.append({
                    "probe_radius_A": probe,
                    "approach_length_A": length,
                    "n_obstacle_atoms": int(obs.n_atoms),
                    "accessible_fraction": float(np.mean(accessible)),
                    "blocked_fraction": float(np.mean(blocked)),
                    "mean_clear_distance_A": float(np.mean(first_dist)),
                    "median_clear_distance_A": float(np.median(first_dist)),
                    "largest_open_cone_half_angle_deg": cone_angle,
                    "largest_open_cone_solid_angle_fraction": cone_fraction,
                    "target_first_block_fraction": float(np.mean(target_first)),
                    "partner_first_block_fraction": float(np.mean(partner_first)),
                    "partner_fraction_among_blocked": partner_among_blocked,
                })
                if (
                    abs(probe - float(primary_probe_radius)) < 1e-8
                    and abs(length - float(primary_approach_length)) < 1e-8
                ):
                    selected_map = accessible.copy()

        if selected_map is None:
            raise RuntimeError("Internal error: primary directional-map setting not evaluated")
        return rows, selected_map


def process_pdb_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    path = str(task["path"])
    ensemble_type = str(task["ensemble_type"])
    prot_chain = str(task["prot_chain"])
    local_directions = np.asarray(task["local_directions"], dtype=np.float64)
    target_residues = list(task["target_residues"])
    probe_radii = list(task["probe_radii"])
    approach_lengths = list(task["approach_lengths"])
    exclusion_windows = list(task["exclusion_windows"])
    primary_window = int(task["primary_window"])
    primary_probe = float(task["primary_probe"])
    primary_length = float(task["primary_length"])
    convergence_counts = list(task["convergence_counts"])
    ray_start = float(task["ray_start"])
    target_mode = str(task["target_mode"])
    gly_target_mode = str(task["gly_target_mode"])
    include_hetero = bool(task["include_hetero"])
    ensemble_weight = float(task["ensemble_weight"])

    atoms = parse_pdb_heavy_atoms(path, include_hetero=include_hetero)
    rows: List[Dict[str, Any]] = []
    maps: Dict[str, np.ndarray] = {}
    convergence_rows: List[Dict[str, Any]] = []

    if ensemble_type == "APO":
        condition_specs = [("APO", [prot_chain])]
    elif ensemble_type == "BOUND":
        partner_chains = parse_partner_chains(str(task["bound_partner_chains"]), atoms, prot_chain)
        condition_specs = [
            ("BOUND_ONLY", [prot_chain]),
            ("BOUND_FULL", [prot_chain] + partner_chains),
        ]
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")

    convergence_direction_sets = {
        int(n): fibonacci_sphere(int(n)) for n in convergence_counts if int(n) != len(local_directions)
    }

    for site in target_residues:
        geom = find_target_geometry(
            atoms, prot_chain, int(site), target_mode, gly_target_mode
        )
        global_directions = local_directions @ geom.frame.T

        for exclude_window in exclusion_windows:
            for condition, chains in condition_specs:
                mask = obstacle_mask(
                    atoms=atoms,
                    included_chains=chains,
                    prot_chain=prot_chain,
                    site=int(site),
                    exclude_window=int(exclude_window),
                )
                obs = atoms.subset(mask)
                condition_rows, selected_map = analyze_condition_site(
                    site_coord=geom.origin,
                    obstacles=obs,
                    global_directions=global_directions,
                    local_directions=local_directions,
                    probe_radii=probe_radii,
                    approach_lengths=approach_lengths,
                    ray_start=ray_start,
                    prot_chain=prot_chain,
                    primary_probe_radius=primary_probe,
                    primary_approach_length=primary_length,
                )
                for row in condition_rows:
                    row.update({
                        "source_pdb": os.path.abspath(path),
                        "source_name": Path(path).name,
                        "ensemble_type": ensemble_type,
                        "condition": condition,
                        "ensemble_weight": ensemble_weight,
                        "prot_chain": prot_chain,
                        "included_chains": ",".join(chains),
                        "target_residue": int(site),
                        "target_resname": geom.resname,
                        "target_one_letter": geom.residue_one_letter,
                        "target_definition": geom.target_definition,
                        "target_atom_name": geom.target_atom_name,
                        "target_atom_names": geom.target_atom_names,
                        "target_atom_count": geom.target_atom_count,
                        "exclude_sequence_window": int(exclude_window),
                        "target_x": float(geom.origin[0]),
                        "target_y": float(geom.origin[1]),
                        "target_z": float(geom.origin[2]),
                        "local_frame_definition": geom.frame_definition,
                    })
                    rows.append(row)

                if int(exclude_window) == primary_window:
                    key = f"{condition}|{int(site)}"
                    maps[key] = np.packbits(selected_map.astype(np.uint8))

                    # Exact deterministic convergence: recompute the primary setting
                    # using independent Fibonacci lattices at each requested direction count.
                    convergence_rows.append({
                        "source_pdb": os.path.abspath(path),
                        "source_name": Path(path).name,
                        "condition": condition,
                        "ensemble_weight": ensemble_weight,
                        "target_residue": int(site),
                        "n_directions": int(len(local_directions)),
                        "accessible_fraction": float(np.mean(selected_map)),
                    })
                    for n_dir, local_dirs_n in convergence_direction_sets.items():
                        global_dirs_n = local_dirs_n @ geom.frame.T
                        _, map_n = analyze_condition_site(
                            site_coord=geom.origin,
                            obstacles=obs,
                            global_directions=global_dirs_n,
                            local_directions=local_dirs_n,
                            probe_radii=[primary_probe],
                            approach_lengths=[primary_length],
                            ray_start=ray_start,
                            prot_chain=prot_chain,
                            primary_probe_radius=primary_probe,
                            primary_approach_length=primary_length,
                        )
                        convergence_rows.append({
                            "source_pdb": os.path.abspath(path),
                            "source_name": Path(path).name,
                            "condition": condition,
                            "ensemble_weight": ensemble_weight,
                            "target_residue": int(site),
                            "n_directions": int(n_dir),
                            "accessible_fraction": float(np.mean(map_n)),
                        })

    return {
        "path": path,
        "ensemble_type": ensemble_type,
        "rows": rows,
        "maps": maps,
        "convergence_rows": convergence_rows,
    }


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(len(weights), 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    w = normalize_weights(weights[valid])
    return float(np.sum(w * values[valid]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    v = values[valid]
    w = normalize_weights(weights[valid])
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    return float(np.interp(float(q), cdf, v))


def effective_sample_size(weights: np.ndarray) -> float:
    w = normalize_weights(np.asarray(weights, dtype=np.float64))
    denom = float(np.sum(w * w))
    return float(1.0 / denom) if denom > 0 else float("nan")


def bootstrap_weighted_mean_ci(
    values: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ) -> Tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        values = values[valid]
        weights = weights[valid]
        if values.size == 0:
            return float("nan"), float("nan")
        if values.size == 1 or n_boot <= 0:
            return float(values[0]), float(values[0])
        p = normalize_weights(weights)
        idx = rng.choice(values.size, size=(n_boot, values.size), replace=True, p=p)
        means = values[idx].mean(axis=1)
        return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_unpaired_weighted_delta_ci(
    a: np.ndarray,
    wa: np.ndarray,
    b: np.ndarray,
    wb: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ) -> Tuple[float, float]:
        a = np.asarray(a, dtype=np.float64)
        wa = np.asarray(wa, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        wb = np.asarray(wb, dtype=np.float64)
        va = np.isfinite(a) & np.isfinite(wa) & (wa > 0)
        vb = np.isfinite(b) & np.isfinite(wb) & (wb > 0)
        a, wa, b, wb = a[va], wa[va], b[vb], wb[vb]
        if a.size == 0 or b.size == 0:
            return float("nan"), float("nan")
        if n_boot <= 0:
            d = weighted_mean(b, wb) - weighted_mean(a, wa)
            return d, d
        ia = rng.choice(a.size, size=(n_boot, a.size), replace=True, p=normalize_weights(wa))
        ib = rng.choice(b.size, size=(n_boot, b.size), replace=True, p=normalize_weights(wb))
        delta = b[ib].mean(axis=1) - a[ia].mean(axis=1)
        return float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))


def bootstrap_paired_weighted_delta_ci(
    delta: np.ndarray,
    weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ) -> Tuple[float, float]:
        delta = np.asarray(delta, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(delta) & np.isfinite(weights) & (weights > 0)
        delta, weights = delta[valid], weights[valid]
        if delta.size == 0:
            return float("nan"), float("nan")
        if delta.size == 1 or n_boot <= 0:
            d = float(delta[0])
            return d, d
        idx = rng.choice(delta.size, size=(n_boot, delta.size), replace=True, p=normalize_weights(weights))
        means = delta[idx].mean(axis=1)
        return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    comparisons = np.sign(b[:, None] - a[None, :])
    return float(np.mean(comparisons))


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    out[np.flatnonzero(valid)] = restored
    return out


def summarize_conditions(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    group_cols = [
        "condition", "target_residue", "probe_radius_A", "approach_length_A",
        "exclude_sequence_window",
    ]
    metrics = [
        "accessible_fraction", "mean_clear_distance_A", "median_clear_distance_A",
        "largest_open_cone_half_angle_deg", "target_first_block_fraction",
        "partner_first_block_fraction", "partner_fraction_among_blocked",
    ]
    for keys, group in df.groupby(group_cols, sort=True):
        base = dict(zip(group_cols, keys))
        weights = group["ensemble_weight"].to_numpy(float)
        base["n_conformers"] = int(len(group))
        base["effective_n"] = effective_sample_size(weights)
        base["weight_sum"] = float(np.sum(weights))
        for metric in metrics:
            vals = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_weighted_mean_ci(vals, weights, n_boot, rng)
            base[f"{metric}_mean"] = weighted_mean(vals, weights)
            base[f"{metric}_median"] = weighted_quantile(vals, weights, 0.5)
            finite = vals[np.isfinite(vals)]
            base[f"{metric}_std_unweighted"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            base[f"{metric}_ci95_low"] = lo
            base[f"{metric}_ci95_high"] = hi
        rows.append(base)
    return pd.DataFrame(rows)


def effect_decomposition(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
    primary_probe: float,
    primary_length: float,
    primary_window: int,
    meaningful_delta: float,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed + 101)
        rows: List[Dict[str, Any]] = []
        settings = (
            df[["target_residue", "probe_radius_A", "approach_length_A", "exclude_sequence_window"]]
            .drop_duplicates()
            .sort_values(["target_residue", "exclude_sequence_window", "probe_radius_A", "approach_length_A"])
        )
        for _, setting in settings.iterrows():
            site = int(setting["target_residue"])
            probe = float(setting["probe_radius_A"])
            length = float(setting["approach_length_A"])
            window = int(setting["exclude_sequence_window"])
            sub = df[
                (df["target_residue"] == site)
                & np.isclose(df["probe_radius_A"], probe)
                & np.isclose(df["approach_length_A"], length)
                & (df["exclude_sequence_window"] == window)
            ]
            apo_df = sub[sub["condition"] == "APO"]
            bo_df = sub[sub["condition"] == "BOUND_ONLY"]
            bf_df = sub[sub["condition"] == "BOUND_FULL"]

            for effect, ref_df, test_df in [
                ("conformational", apo_df, bo_df),
                ("total", apo_df, bf_df),
            ]:
                ref = ref_df["accessible_fraction"].to_numpy(float)
                wr = ref_df["ensemble_weight"].to_numpy(float)
                test = test_df["accessible_fraction"].to_numpy(float)
                wt = test_df["ensemble_weight"].to_numpy(float)
                ref_mean = weighted_mean(ref, wr)
                test_mean = weighted_mean(test, wt)
                delta = test_mean - ref_mean
                lo, hi = bootstrap_unpaired_weighted_delta_ci(ref, wr, test, wt, n_boot, rng)
                uniform = (
                    np.allclose(wr, wr[0]) if len(wr) else True
                ) and (
                    np.allclose(wt, wt[0]) if len(wt) else True
                )
                if uniform and len(ref) and len(test):
                    try:
                        pvalue = float(mannwhitneyu(ref, test, alternative="two-sided").pvalue)
                        test_name = "Mann-Whitney U (equal-weight ensembles)"
                    except Exception:
                        pvalue, test_name = float("nan"), "not available"
                else:
                    pvalue, test_name = float("nan"), "descriptive weighted bootstrap"
                rows.append({
                    "target_residue": site,
                    "probe_radius_A": probe,
                    "approach_length_A": length,
                    "exclude_sequence_window": window,
                    "is_primary_setting": bool(
                        abs(probe-primary_probe)<1e-8 and abs(length-primary_length)<1e-8
                        and window == primary_window
                    ),
                    "effect": effect,
                    "reference_condition": str(ref_df["condition"].iloc[0]) if len(ref_df) else "APO",
                    "test_condition": str(test_df["condition"].iloc[0]) if len(test_df) else "",
                    "reference_mean_accessible_fraction": ref_mean,
                    "test_mean_accessible_fraction": test_mean,
                    "delta_accessible_fraction": delta,
                    "relative_change_percent": (
                        100.0 * delta / ref_mean if np.isfinite(ref_mean) and ref_mean > 1e-12 else float("nan")
                    ),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "n_reference": int(len(ref)),
                    "n_test": int(len(test)),
                    "effective_n_reference": effective_sample_size(wr),
                    "effective_n_test": effective_sample_size(wt),
                    "test_name": test_name,
                    "pvalue": pvalue,
                    "effect_size_name": "Cliff_delta_test_minus_reference_unweighted",
                    "effect_size": cliffs_delta(ref, test),
                    "meaningful_loss_threshold": meaningful_delta,
                    "fraction_test_below_reference_not_paired": float("nan"),
                    "fraction_meaningfully_masked": float("nan"),
                })

            paired = bo_df[["source_pdb", "accessible_fraction", "ensemble_weight"]].merge(
                bf_df[["source_pdb", "accessible_fraction"]],
                on="source_pdb", suffixes=("_bound_only", "_bound_full"), how="inner",
            )
            delta = (
                paired["accessible_fraction_bound_full"].to_numpy(float)
                - paired["accessible_fraction_bound_only"].to_numpy(float)
            )
            weights = paired["ensemble_weight"].to_numpy(float)
            mean_delta = weighted_mean(delta, weights)
            lo, hi = bootstrap_paired_weighted_delta_ci(delta, weights, n_boot, rng)
            meaningful = delta <= -float(meaningful_delta)
            any_loss = delta < -1e-12
            rows.append({
                "target_residue": site,
                "probe_radius_A": probe,
                "approach_length_A": length,
                "exclude_sequence_window": window,
                "is_primary_setting": bool(
                    abs(probe-primary_probe)<1e-8 and abs(length-primary_length)<1e-8
                    and window == primary_window
                ),
                "effect": "direct_partner",
                "reference_condition": "BOUND_ONLY",
                "test_condition": "BOUND_FULL",
                "reference_mean_accessible_fraction": weighted_mean(
                    paired["accessible_fraction_bound_only"].to_numpy(float), weights
                ),
                "test_mean_accessible_fraction": weighted_mean(
                    paired["accessible_fraction_bound_full"].to_numpy(float), weights
                ),
                "delta_accessible_fraction": mean_delta,
                "relative_change_percent": (
                    100.0 * mean_delta / weighted_mean(
                        paired["accessible_fraction_bound_only"].to_numpy(float), weights
                    ) if weighted_mean(
                        paired["accessible_fraction_bound_only"].to_numpy(float), weights
                    ) > 1e-12 else float("nan")
                ),
                "delta_ci95_low": lo,
                "delta_ci95_high": hi,
                "n_reference": int(len(paired)),
                "n_test": int(len(paired)),
                "effective_n_reference": effective_sample_size(weights),
                "effective_n_test": effective_sample_size(weights),
                "test_name": "descriptive paired weighted bootstrap",
                "pvalue": float("nan"),
                "effect_size_name": "paired_weighted_mean_delta",
                "effect_size": mean_delta,
                "meaningful_loss_threshold": meaningful_delta,
                "fraction_test_below_reference_not_paired": weighted_mean(any_loss.astype(float), weights),
                "fraction_meaningfully_masked": weighted_mean(meaningful.astype(float), weights),
            })

        out = pd.DataFrame(rows)
        out["qvalue_bh_exploratory"] = np.nan
        mask_test = out["effect"].isin(["conformational", "total"]) & out["pvalue"].notna()
        out.loc[mask_test, "qvalue_bh_exploratory"] = benjamini_hochberg(
            out.loc[mask_test, "pvalue"].to_numpy(float)
        )
        out["qvalue_bh_primary_family"] = np.nan
        primary_mask = mask_test & out["is_primary_setting"]
        out.loc[primary_mask, "qvalue_bh_primary_family"] = benjamini_hochberg(
            out.loc[primary_mask, "pvalue"].to_numpy(float)
        )
        return out


def safe_asymmetric_yerr(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    ) -> np.ndarray:
        lower = np.maximum(0.0, y - lo)
        upper = np.maximum(0.0, hi - y)
        return np.vstack([lower, upper])


def _primary_subset(df: pd.DataFrame, probe: float, length: float, window: int) -> pd.DataFrame:
    return df[
        np.isclose(df["probe_radius_A"], probe)
        & np.isclose(df["approach_length_A"], length)
        & (df["exclude_sequence_window"] == int(window))
    ].copy()


def build_residue_one_letter_map(per_df: pd.DataFrame) -> Dict[int, str]:
    required = {"target_residue", "target_one_letter"}
    missing = required - set(per_df.columns)
    if missing:
        raise ValueError(
            "per_conformer_accessibility.csv is missing columns required for residue labels: "
            f"{sorted(missing)}"
        )
    mapping: Dict[int, str] = {}
    for site, group in per_df.groupby("target_residue", sort=True):
        identities = sorted({str(x).strip().upper() for x in group["target_one_letter"] if str(x).strip()})
        if len(identities) != 1:
            raise ValueError(
                f"Target residue {int(site)} has inconsistent one-letter identities: {identities}"
            )
        mapping[int(site)] = identities[0]
    return mapping


def format_residue_plot_label(
    site: int,
    residue_one_letters: Mapping[int, str],
    residue_label_shift: int,
    ) -> str:
        identity = str(residue_one_letters.get(int(site), "X")).strip().upper() or "X"
        return f"{identity}{int(site) + int(residue_label_shift)}"


def residue_plot_labels(
    sites: Sequence[int],
    residue_one_letters: Mapping[int, str],
    residue_label_shift: int,
    ) -> List[str]:
        return [
            format_residue_plot_label(int(site), residue_one_letters, residue_label_shift)
            for site in sites
        ]


def plot_summary(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    outdir: Path,
    probe: float,
    length: float,
    window: int,
    meaningful_delta: float,
    dpi: int,
    residue_one_letters: Mapping[int, str],
    residue_label_shift: int,
    ) -> pd.DataFrame:
        sub = _primary_subset(summary, probe, length, window)
        eff = _primary_subset(effects, probe, length, window)
        sites = sorted(int(x) for x in sub["target_residue"].unique())
        labels = residue_plot_labels(sites, residue_one_letters, residue_label_shift)
        x = np.arange(len(sites), dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
        ax = axes[0, 0]
        width = 0.25
        for idx, condition in enumerate(CONDITION_ORDER):
            g = sub[sub["condition"] == condition].set_index("target_residue").reindex(sites)
            y = g["accessible_fraction_mean"].to_numpy(float)
            lo = g["accessible_fraction_ci95_low"].to_numpy(float)
            hi = g["accessible_fraction_ci95_high"].to_numpy(float)
            ax.bar(
                x + (idx - 1) * width, y, width=width,
                yerr=safe_asymmetric_yerr(y, lo, hi), capsize=3,
                label=CONDITION_LABEL[condition],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target residue")
        ax.set_ylabel("Accessible Vector Yield Fraction")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=9)
        ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontweight="bold", fontsize=14)

        ax = axes[0, 1]
        effects_order = ["conformational", "direct_partner", "total"]
        effect_labels = [
            "Bound-state − APO",
            "Bound-full − Bound-state",
            "Bound-full − APO",
        ]
        width = 0.25
        for idx, (effect_name, label) in enumerate(zip(effects_order, effect_labels)):
            g = eff[eff["effect"] == effect_name].set_index("target_residue").reindex(sites)
            y = g["delta_accessible_fraction"].to_numpy(float)
            lo = g["delta_ci95_low"].to_numpy(float)
            hi = g["delta_ci95_high"].to_numpy(float)
            ax.bar(
                x + (idx - 1) * width, y, width=width,
                yerr=safe_asymmetric_yerr(y, lo, hi), capsize=3, label=label,
            )
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Target residue")
        ax.set_ylabel("Δ Accessible Vector Yield Fraction")
        ax.legend(frameon=False, fontsize=8)
        ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontweight="bold", fontsize=14)

        ax = axes[1, 0]
        direct = eff[eff["effect"] == "direct_partner"].set_index("target_residue").reindex(sites)
        prevalence = direct["fraction_meaningfully_masked"].to_numpy(float)
        ax.bar(x, prevalence)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Target residue")
        ax.set_ylabel("Fraction of bound conformers")
        ax.set_title(f"Directly masked by ≥ {meaningful_delta:.3f} accessible fraction")
        ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontweight="bold", fontsize=14)

        ax = axes[1, 1]
        total = effects[
            (effects["effect"] == "total")
            & (effects["exclude_sequence_window"] == int(window))
            & np.isclose(effects["approach_length_A"], length)
        ].copy()
        probes = sorted(total["probe_radius_A"].unique())
        matrix = np.full((len(sites), len(probes)), np.nan)
        for i, site in enumerate(sites):
            for j, pr in enumerate(probes):
                row = total[(total["target_residue"] == site) & np.isclose(total["probe_radius_A"], pr)]
                if len(row):
                    matrix[i, j] = float(row.iloc[0]["delta_accessible_fraction"])
        max_abs = float(np.nanmax(np.abs(matrix))) if np.any(np.isfinite(matrix)) else 1.0
        max_abs = max(max_abs, 1e-6)
        im = ax.imshow(matrix, aspect="auto", vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(np.arange(len(probes)))
        ax.set_xticklabels([f"{p:g}" for p in probes])
        ax.set_yticks(np.arange(len(sites)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Probe radius (Å)")
        ax.set_ylabel("Target residue")
        ax.set_title(f"Total bound effect at {length:g} Å path")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Bound-full − APO")
        ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontweight="bold", fontsize=14)

        fig.suptitle(
            f"Target residue approach accessibility\n"
            f"Primary probe {probe:g} Å; path {length:g} Å; local exclusion ±{window} residues",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outdir / "primary_summary.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        rows = []
        for site in sites:
            row: Dict[str, Any] = {
                "target_residue": int(site),
                "plot_residue_label": format_residue_plot_label(
                    int(site), residue_one_letters, residue_label_shift
                ),
                "plot_residue_label_shift": int(residue_label_shift),
                "primary_probe_radius_A": probe,
                "primary_approach_length_A": length,
                "primary_exclude_sequence_window": window,
            }
            for condition in CONDITION_ORDER:
                g = sub[(sub["condition"] == condition) & (sub["target_residue"] == site)]
                if len(g):
                    r = g.iloc[0]
                    row[f"{condition}_mean_accessible_fraction"] = r["accessible_fraction_mean"]
                    row[f"{condition}_ci95_low"] = r["accessible_fraction_ci95_low"]
                    row[f"{condition}_ci95_high"] = r["accessible_fraction_ci95_high"]
            for effect_name in effects_order:
                g = eff[(eff["effect"] == effect_name) & (eff["target_residue"] == site)]
                if len(g):
                    r = g.iloc[0]
                    row[f"{effect_name}_delta"] = r["delta_accessible_fraction"]
                    row[f"{effect_name}_ci95_low"] = r["delta_ci95_low"]
                    row[f"{effect_name}_ci95_high"] = r["delta_ci95_high"]
                    row[f"{effect_name}_relative_change_percent"] = r["relative_change_percent"]
                    if effect_name == "direct_partner":
                        row["direct_partner_fraction_meaningfully_masked"] = r["fraction_meaningfully_masked"]
                        row["direct_partner_fraction_any_loss"] = r["fraction_test_below_reference_not_paired"]
            rows.append(row)
        return pd.DataFrame(rows)


def directions_to_lon_lat(directions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.arctan2(directions[:, 1], directions[:, 0])
    lat = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
    return lon, lat


def plot_directional_maps(
    map_means: Mapping[Tuple[str, int], np.ndarray],
    local_directions: np.ndarray,
    target_residues: Sequence[int],
    outdir: Path,
    probe: float,
    length: float,
    window: int,
    dpi: int,
    residue_one_letters: Mapping[int, str],
    residue_label_shift: int,
    ) -> None:
        lon, lat = directions_to_lon_lat(local_directions)
        for site in target_residues:
            site_label = format_residue_plot_label(
                int(site), residue_one_letters, residue_label_shift
            )
            fig = plt.figure(figsize=(15, 4.8))
            for idx, condition in enumerate(CONDITION_ORDER, start=1):
                ax = fig.add_subplot(1, 3, idx, projection="mollweide")
                vals = map_means[(condition, int(site))]
                sc = ax.scatter(lon, lat, c=vals, s=8, vmin=0.0, vmax=1.0, rasterized=True)
                ax.set_title(CONDITION_LABEL[condition])
                ax.grid(True, alpha=0.3)
            cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.72, pad=0.05)
            cbar.set_label("Fraction of conformers accessible")
            fig.suptitle(
                f"Local-frame directional accessibility: {site_label}\n"
                f"probe {probe:g} Å; path {length:g} Å; exclusion ±{window}", y=1.03,
            )
            fig.subplots_adjust(left=0.03, right=0.90, bottom=0.08, top=0.84, wspace=0.16)
            fig.savefig(outdir / f"directional_accessibility_local_site_{site}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            apo = map_means[("APO", int(site))]
            bound_only = map_means[("BOUND_ONLY", int(site))]
            bound_full = map_means[("BOUND_FULL", int(site))]
            delta_maps = [
                (bound_only - apo, "Bound-state − APO"),
                (bound_full - bound_only, "Bound-full − Bound-state"),
                (bound_full - apo, "Bound-full − APO"),
            ]
            max_abs = max(float(np.nanmax(np.abs(v))) for v, _ in delta_maps)
            max_abs = max(max_abs, 1e-6)
            fig = plt.figure(figsize=(15, 4.8))
            for idx, (vals, title) in enumerate(delta_maps, start=1):
                ax = fig.add_subplot(1, 3, idx, projection="mollweide")
                sc = ax.scatter(lon, lat, c=vals, s=8, vmin=-max_abs, vmax=max_abs, rasterized=True)
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.3)
            cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.72, pad=0.05)
            cbar.set_label("Δ fraction of conformers accessible")
            fig.suptitle(
                f"Local-frame directional effects: {site_label}\nnegative = reduced access",
                y=1.03,
            )
            fig.subplots_adjust(left=0.03, right=0.90, bottom=0.08, top=0.84, wspace=0.16)
            fig.savefig(outdir / f"directional_effects_local_site_{site}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def summarize_direction_convergence(
    convergence_df: pd.DataFrame,
    full_n: int,
    ) -> pd.DataFrame:
        full = convergence_df[convergence_df["n_directions"] == int(full_n)][
            ["source_pdb", "condition", "target_residue", "accessible_fraction"]
        ].rename(columns={"accessible_fraction": "accessible_fraction_full"})
        merged = convergence_df.merge(full, on=["source_pdb", "condition", "target_residue"], how="left")
        merged["absolute_error_vs_full"] = np.abs(
            merged["accessible_fraction"] - merged["accessible_fraction_full"]
        )
        rows = []
        for keys, group in merged.groupby(["condition", "target_residue", "n_directions"], sort=True):
            condition, site, n_dir = keys
            weights = group["ensemble_weight"].to_numpy(float)
            err = group["absolute_error_vs_full"].to_numpy(float)
            rows.append({
                "condition": condition,
                "target_residue": int(site),
                "n_directions": int(n_dir),
                "weighted_mean_absolute_error_vs_full": weighted_mean(err, weights),
                "max_absolute_error_vs_full": float(np.nanmax(err)),
                "weighted_mean_accessible_fraction": weighted_mean(
                    group["accessible_fraction"].to_numpy(float), weights
                ),
            })
        return pd.DataFrame(rows)


def plot_direction_convergence(summary: pd.DataFrame, outdir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for condition in CONDITION_ORDER:
        g = summary[summary["condition"] == condition].groupby("n_directions", as_index=False)[
            "weighted_mean_absolute_error_vs_full"
        ].mean()
        ax.plot(g["n_directions"], g["weighted_mean_absolute_error_vs_full"], marker="o", label=CONDITION_LABEL[condition])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of sampled directions")
    ax.set_ylabel("Mean absolute error vs full direction set")
    ax.set_title("Directional-sampling convergence at primary parameters")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "direction_convergence_primary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_exclusion_window_sensitivity(
    summary: pd.DataFrame,
    outdir: Path,
    probe: float,
    length: float,
    dpi: int,
    residue_one_letters: Mapping[int, str],
    residue_label_shift: int,
    ) -> pd.DataFrame:
        sub = summary[
            np.isclose(summary["probe_radius_A"], probe)
            & np.isclose(summary["approach_length_A"], length)
        ].copy()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for condition in CONDITION_ORDER:
            for site in sorted(int(x) for x in sub["target_residue"].unique()):
                g = sub[(sub["condition"] == condition) & (sub["target_residue"] == site)].sort_values("exclude_sequence_window")
                site_label = format_residue_plot_label(
                    site, residue_one_letters, residue_label_shift
                )
                ax.plot(
                    g["exclude_sequence_window"], g["accessible_fraction_mean"],
                    marker="o", alpha=0.7,
                    label=f"{CONDITION_LABEL[condition]} {site_label}",
                )
        ax.set_xlabel("Excluded local sequence half-window (residues)")
        ax.set_ylabel("Mean Accessible Vector Yield Fraction")
        ax.set_title(f"Local-exclusion sensitivity: probe {probe:g} Å, path {length:g} Å")
        ax.legend(frameon=False, fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(outdir / "exclusion_window_sensitivity_primary.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return sub


def load_weight_map(csv_path: Optional[str], files: Sequence[str]) -> Dict[str, float]:
    names = [Path(f).name for f in files]
    if csv_path is None:
        return {name: 1.0 for name in names}
    table = pd.read_csv(csv_path)
    required = {"filename", "weight"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Weights CSV {csv_path} missing columns: {sorted(missing)}")
    mapping = {str(r.filename): float(r.weight) for r in table.itertuples(index=False)}
    absent = [name for name in names if name not in mapping]
    if absent:
        raise ValueError(f"Weights CSV {csv_path} lacks {len(absent)} input files; examples: {absent[:5]}")
    invalid = [name for name in names if not np.isfinite(mapping[name]) or mapping[name] <= 0]
    if invalid:
        raise ValueError(f"Weights must be positive finite values; invalid examples: {invalid[:5]}")
    return {name: mapping[name] for name in names}



REGION_ENDPOINTS = [
    "region_mean_accessible_fraction",
    "region_min_accessible_fraction",
    "region_median_accessible_fraction",
]


def build_region_per_conformer(
    df: pd.DataFrame,
    region_name: str,
    region_residues: Sequence[int],
    ) -> pd.DataFrame:
        """Aggregate residue-level accessibility within each conformer before inference."""
        residues = sorted(set(int(x) for x in region_residues))
        wanted = df[df["target_residue"].isin(residues)].copy()
        group_cols = [
            "source_pdb", "source_name", "ensemble_type", "condition", "ensemble_weight",
            "prot_chain", "included_chains", "probe_radius_A", "approach_length_A",
            "exclude_sequence_window",
        ]
        rows: List[Dict[str, Any]] = []
        for keys, group in wanted.groupby(group_cols, sort=True, dropna=False):
            observed = sorted(set(int(x) for x in group["target_residue"]))
            if observed != residues:
                raise RuntimeError(
                    f"Region {region_name!r} is incomplete for {group.iloc[0]['source_name']} "
                    f"condition {group.iloc[0]['condition']}: expected {residues}, observed {observed}"
                )
            values = group.sort_values("target_residue")["accessible_fraction"].to_numpy(float)
            base = dict(zip(group_cols, keys))
            base.update({
                "region_name": str(region_name),
                "region_residues": ",".join(map(str, residues)),
                "n_region_residues": int(len(residues)),
                "region_mean_accessible_fraction": float(np.mean(values)),
                "region_min_accessible_fraction": float(np.min(values)),
                "region_median_accessible_fraction": float(np.median(values)),
            })
            rows.append(base)
        return pd.DataFrame(rows)


def summarize_region_conditions(
    region_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        rng = np.random.default_rng(seed + 401)
        group_cols = [
            "region_name", "region_residues", "condition", "probe_radius_A",
            "approach_length_A", "exclude_sequence_window",
        ]
        for keys, group in region_df.groupby(group_cols, sort=True):
            base = dict(zip(group_cols, keys))
            weights = group["ensemble_weight"].to_numpy(float)
            base["n_conformers"] = int(len(group))
            base["effective_n"] = effective_sample_size(weights)
            base["weight_sum"] = float(np.sum(weights))
            for endpoint in REGION_ENDPOINTS:
                values = group[endpoint].to_numpy(float)
                lo, hi = bootstrap_weighted_mean_ci(values, weights, n_boot, rng)
                base[f"{endpoint}_mean"] = weighted_mean(values, weights)
                base[f"{endpoint}_median"] = weighted_quantile(values, weights, 0.5)
                finite = values[np.isfinite(values)]
                base[f"{endpoint}_std_unweighted"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                )
                base[f"{endpoint}_ci95_low"] = lo
                base[f"{endpoint}_ci95_high"] = hi
            rows.append(base)
        return pd.DataFrame(rows)


def region_effect_decomposition(
    region_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    primary_probe: float,
    primary_length: float,
    primary_window: int,
    meaningful_delta: float,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed + 503)
        rows: List[Dict[str, Any]] = []
        settings = (
            region_df[["probe_radius_A", "approach_length_A", "exclude_sequence_window"]]
            .drop_duplicates()
            .sort_values(["exclude_sequence_window", "probe_radius_A", "approach_length_A"])
        )
        for endpoint in REGION_ENDPOINTS:
            for _, setting in settings.iterrows():
                probe = float(setting["probe_radius_A"])
                length = float(setting["approach_length_A"])
                window = int(setting["exclude_sequence_window"])
                sub = region_df[
                    np.isclose(region_df["probe_radius_A"], probe)
                    & np.isclose(region_df["approach_length_A"], length)
                    & (region_df["exclude_sequence_window"] == window)
                ]
                apo_df = sub[sub["condition"] == "APO"]
                bo_df = sub[sub["condition"] == "BOUND_ONLY"]
                bf_df = sub[sub["condition"] == "BOUND_FULL"]
                common_meta = {
                    "region_name": str(sub["region_name"].iloc[0]),
                    "region_residues": str(sub["region_residues"].iloc[0]),
                    "endpoint": endpoint,
                    "probe_radius_A": probe,
                    "approach_length_A": length,
                    "exclude_sequence_window": window,
                    "is_primary_setting": bool(
                        abs(probe - primary_probe) < 1e-8
                        and abs(length - primary_length) < 1e-8
                        and window == primary_window
                    ),
                }
                for effect, ref_df, test_df in [
                    ("conformational", apo_df, bo_df),
                    ("total", apo_df, bf_df),
                ]:
                    ref = ref_df[endpoint].to_numpy(float)
                    wr = ref_df["ensemble_weight"].to_numpy(float)
                    test = test_df[endpoint].to_numpy(float)
                    wt = test_df["ensemble_weight"].to_numpy(float)
                    ref_mean = weighted_mean(ref, wr)
                    test_mean = weighted_mean(test, wt)
                    delta = test_mean - ref_mean
                    lo, hi = bootstrap_unpaired_weighted_delta_ci(ref, wr, test, wt, n_boot, rng)
                    uniform = (
                        np.allclose(wr, wr[0]) if len(wr) else True
                    ) and (
                        np.allclose(wt, wt[0]) if len(wt) else True
                    )
                    if uniform and len(ref) and len(test):
                        try:
                            pvalue = float(mannwhitneyu(ref, test, alternative="two-sided").pvalue)
                            test_name = "Mann-Whitney U (equal-weight ensembles)"
                        except Exception:
                            pvalue, test_name = float("nan"), "not available"
                    else:
                        pvalue, test_name = float("nan"), "descriptive weighted bootstrap"
                    rows.append({
                        **common_meta,
                        "effect": effect,
                        "reference_condition": str(ref_df["condition"].iloc[0]) if len(ref_df) else "APO",
                        "test_condition": str(test_df["condition"].iloc[0]) if len(test_df) else "",
                        "reference_mean": ref_mean,
                        "test_mean": test_mean,
                        "delta": delta,
                        "relative_change_percent": (
                            100.0 * delta / ref_mean
                            if np.isfinite(ref_mean) and ref_mean > 1e-12 else float("nan")
                        ),
                        "delta_ci95_low": lo,
                        "delta_ci95_high": hi,
                        "n_reference": int(len(ref)),
                        "n_test": int(len(test)),
                        "effective_n_reference": effective_sample_size(wr),
                        "effective_n_test": effective_sample_size(wt),
                        "test_name": test_name,
                        "pvalue": pvalue,
                        "effect_size_name": "Cliff_delta_test_minus_reference_unweighted",
                        "effect_size": cliffs_delta(ref, test),
                        "meaningful_loss_threshold": meaningful_delta,
                        "fraction_any_loss": float("nan"),
                        "fraction_meaningfully_masked": float("nan"),
                    })

                paired = bo_df[["source_pdb", endpoint, "ensemble_weight"]].merge(
                    bf_df[["source_pdb", endpoint]],
                    on="source_pdb", suffixes=("_bound_only", "_bound_full"), how="inner",
                )
                bound_only_col = f"{endpoint}_bound_only"
                bound_full_col = f"{endpoint}_bound_full"
                delta_values = (
                    paired[bound_full_col].to_numpy(float)
                    - paired[bound_only_col].to_numpy(float)
                )
                weights = paired["ensemble_weight"].to_numpy(float)
                mean_delta = weighted_mean(delta_values, weights)
                lo, hi = bootstrap_paired_weighted_delta_ci(delta_values, weights, n_boot, rng)
                reference_mean = weighted_mean(paired[bound_only_col].to_numpy(float), weights)
                rows.append({
                    **common_meta,
                    "effect": "direct_partner",
                    "reference_condition": "BOUND_ONLY",
                    "test_condition": "BOUND_FULL",
                    "reference_mean": reference_mean,
                    "test_mean": weighted_mean(paired[bound_full_col].to_numpy(float), weights),
                    "delta": mean_delta,
                    "relative_change_percent": (
                        100.0 * mean_delta / reference_mean if reference_mean > 1e-12 else float("nan")
                    ),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "n_reference": int(len(paired)),
                    "n_test": int(len(paired)),
                    "effective_n_reference": effective_sample_size(weights),
                    "effective_n_test": effective_sample_size(weights),
                    "test_name": "descriptive paired weighted bootstrap",
                    "pvalue": float("nan"),
                    "effect_size_name": "paired_weighted_mean_delta",
                    "effect_size": mean_delta,
                    "meaningful_loss_threshold": meaningful_delta,
                    "fraction_any_loss": weighted_mean((delta_values < -1e-12).astype(float), weights),
                    "fraction_meaningfully_masked": weighted_mean(
                        (delta_values <= -float(meaningful_delta)).astype(float), weights
                    ),
                })

        out = pd.DataFrame(rows)
        out["qvalue_bh_exploratory"] = np.nan
        test_mask = out["effect"].isin(["conformational", "total"]) & out["pvalue"].notna()
        out.loc[test_mask, "qvalue_bh_exploratory"] = benjamini_hochberg(
            out.loc[test_mask, "pvalue"].to_numpy(float)
        )
        out["qvalue_bh_primary_family"] = np.nan
        primary_mask = test_mask & out["is_primary_setting"]
        out.loc[primary_mask, "qvalue_bh_primary_family"] = benjamini_hochberg(
            out.loc[primary_mask, "pvalue"].to_numpy(float)
        )
        return out


def plot_region_primary_summary(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    outdir: Path,
    probe: float,
    length: float,
    window: int,
    dpi: int,
    ) -> pd.DataFrame:
        endpoints = ["region_mean_accessible_fraction", "region_min_accessible_fraction"]
        endpoint_labels = ["Region mean", "Region minimum"]
        sub = _primary_subset(summary, probe, length, window)
        eff = _primary_subset(effects, probe, length, window)
        x = np.arange(len(endpoints), dtype=float)
        width = 0.25
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        ax = axes[0]
        for idx, condition in enumerate(CONDITION_ORDER):
            row = sub[sub["condition"] == condition]
            y = np.asarray([float(row.iloc[0][f"{e}_mean"]) for e in endpoints])
            lo = np.asarray([float(row.iloc[0][f"{e}_ci95_low"]) for e in endpoints])
            hi = np.asarray([float(row.iloc[0][f"{e}_ci95_high"]) for e in endpoints])
            ax.bar(
                x + (idx - 1) * width, y, width=width,
                yerr=safe_asymmetric_yerr(y, lo, hi), capsize=3,
                label=CONDITION_LABEL[condition],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(endpoint_labels)
        ax.set_ylabel("Accessible Vector Yield Fraction")
        ax.set_ylim(bottom=0.0)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Region accessibility")

        ax = axes[1]
        effect_order = ["conformational", "direct_partner", "total"]
        effect_labels = [
            "Bound-state − APO",
            "Bound-full − Bound-state",
            "Bound-full − APO",
        ]
        for idx, (effect, label) in enumerate(zip(effect_order, effect_labels)):
            ys, los, his = [], [], []
            for endpoint in endpoints:
                row = eff[(eff["effect"] == effect) & (eff["endpoint"] == endpoint)].iloc[0]
                ys.append(float(row["delta"]))
                los.append(float(row["delta_ci95_low"]))
                his.append(float(row["delta_ci95_high"]))
            y = np.asarray(ys); lo = np.asarray(los); hi = np.asarray(his)
            ax.bar(
                x + (idx - 1) * width, y, width=width,
                yerr=safe_asymmetric_yerr(y, lo, hi), capsize=3, label=label,
            )
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(endpoint_labels)
        ax.set_ylabel("Δ Accessible Vector Yield Fraction")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Region effects")

        region_name = str(sub.iloc[0]["region_name"])
        fig.suptitle(
            f"{region_name} region approach accessibility\n"
            f"probe {probe:g} Å; path {length:g} Å; local exclusion ±{window} residues"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(outdir / "region_summary_primary.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        rows: List[Dict[str, Any]] = []
        for endpoint in REGION_ENDPOINTS:
            row: Dict[str, Any] = {
                "region_name": region_name,
                "region_residues": str(sub.iloc[0]["region_residues"]),
                "endpoint": endpoint,
                "primary_probe_radius_A": probe,
                "primary_approach_length_A": length,
                "primary_exclude_sequence_window": window,
            }
            for condition in CONDITION_ORDER:
                g = sub[sub["condition"] == condition].iloc[0]
                row[f"{condition}_mean"] = g[f"{endpoint}_mean"]
                row[f"{condition}_ci95_low"] = g[f"{endpoint}_ci95_low"]
                row[f"{condition}_ci95_high"] = g[f"{endpoint}_ci95_high"]
            for effect in ["conformational", "direct_partner", "total"]:
                g = eff[(eff["effect"] == effect) & (eff["endpoint"] == endpoint)].iloc[0]
                row[f"{effect}_delta"] = g["delta"]
                row[f"{effect}_ci95_low"] = g["delta_ci95_low"]
                row[f"{effect}_ci95_high"] = g["delta_ci95_high"]
                row[f"{effect}_relative_change_percent"] = g["relative_change_percent"]
                if effect == "direct_partner":
                    row["direct_partner_fraction_meaningfully_masked"] = g[
                        "fraction_meaningfully_masked"
                    ]
                    row["direct_partner_fraction_any_loss"] = g["fraction_any_loss"]
            rows.append(row)
        return pd.DataFrame(rows)


def write_run_config(args: argparse.Namespace, outdir: Path) -> None:
    config = vars(args).copy()
    for key in ["target_residues", "probe_radii", "approach_lengths", "exclude_sequence_windows", "direction_convergence_counts"]:
        config[key] = list(config[key])
    if config.get("region_residues") is not None:
        config["region_residues"] = list(config["region_residues"])
    with open(outdir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def _read_required_csv(source_dir: Path, filename: str) -> pd.DataFrame:
    path = source_dir / filename
    if not path.is_file():
        raise SystemExit(f"Plot-only mode requires {path}")
    return pd.read_csv(path)


def _load_previous_run_config(source_dir: Path) -> Dict[str, Any]:
    path = source_dir / "run_config.json"
    if not path.is_file():
        raise SystemExit(f"Plot-only mode requires {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    required = [
        "primary_probe_radius", "primary_approach_length",
        "primary_exclude_window", "minimum_meaningful_delta",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"{path} is missing required settings: {missing}")
    return config


def _load_directional_maps_for_redo(
    npz_path: Path,
    target_residues: Sequence[int],
    ) -> Tuple[np.ndarray, Dict[Tuple[str, int], np.ndarray]]:
        with np.load(npz_path, allow_pickle=False) as payload:
            if "local_directions_xyz" not in payload:
                raise SystemExit(f"{npz_path} lacks local_directions_xyz")
            local_directions = np.asarray(payload["local_directions_xyz"], dtype=np.float64)
            map_means: Dict[Tuple[str, int], np.ndarray] = {}
            missing_keys: List[str] = []
            for condition in CONDITION_ORDER:
                for site in target_residues:
                    key = f"{condition}_site_{int(site)}"
                    if key not in payload:
                        missing_keys.append(key)
                    else:
                        map_means[(condition, int(site))] = np.asarray(payload[key], dtype=np.float64)
        if missing_keys:
            raise SystemExit(
                f"{npz_path} is missing directional map arrays: {missing_keys[:10]}"
            )
        return local_directions, map_means


def redo_plots_from_existing_output(args: argparse.Namespace) -> None:
    source_dir = Path(args.redo_plots).expanduser().resolve()
    outdir = Path(args.outdir).expanduser()
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = _load_previous_run_config(source_dir)
    probe = float(config["primary_probe_radius"])
    length = float(config["primary_approach_length"])
    window = int(config["primary_exclude_window"])
    meaningful_delta = float(config["minimum_meaningful_delta"])

    per_df = _read_required_csv(source_dir, "per_conformer_accessibility.csv")
    residue_one_letters = build_residue_one_letter_map(per_df)
    condition_summary = _read_required_csv(source_dir, "condition_summary.csv")
    effects = _read_required_csv(source_dir, "accessibility_effect_decomposition.csv")
    target_residues = sorted(int(x) for x in condition_summary["target_residue"].unique())

    primary_summary = plot_summary(
        condition_summary, effects, plots_dir,
        probe, length, window, meaningful_delta, args.figure_dpi,
        residue_one_letters, args.residue_label_shift,
    )
    primary_summary.to_csv(outdir / "primary_summary.csv", index=False)

    region_summary_path = source_dir / "region_condition_summary.csv"
    region_effects_path = source_dir / "region_effect_decomposition.csv"
    if region_summary_path.is_file() or region_effects_path.is_file():
        if not (region_summary_path.is_file() and region_effects_path.is_file()):
            raise SystemExit(
                "Incomplete prior region outputs: both region_condition_summary.csv and "
                "region_effect_decomposition.csv are required to redo the region plot."
            )
        region_summary = pd.read_csv(region_summary_path)
        region_effects = pd.read_csv(region_effects_path)
        region_primary = plot_region_primary_summary(
            region_summary, region_effects, plots_dir,
            probe, length, window, args.figure_dpi,
        )
        region_primary.to_csv(outdir / "region_primary_summary.csv", index=False)

    convergence_path = source_dir / "direction_convergence_summary.csv"
    if convergence_path.is_file():
        convergence_summary = pd.read_csv(convergence_path)
        plot_direction_convergence(convergence_summary, plots_dir, args.figure_dpi)

    exclusion_summary = plot_exclusion_window_sensitivity(
        condition_summary, plots_dir, probe, length, args.figure_dpi,
        residue_one_letters, args.residue_label_shift,
    )
    exclusion_summary.to_csv(outdir / "exclusion_window_sensitivity.csv", index=False)

    directional_path = source_dir / "directional_accessibility_maps_local_frame.npz"
    if args.save_directional_maps and directional_path.is_file():
        local_directions, map_means = _load_directional_maps_for_redo(
            directional_path, target_residues
        )
        plot_directional_maps(
            map_means, local_directions, target_residues, plots_dir,
            probe, length, window, args.figure_dpi,
            residue_one_letters, args.residue_label_shift,
        )

    redo_config = {
        "mode": "redo_plots",
        "source_output_dir": str(source_dir),
        "new_output_dir": str(outdir.resolve()),
        "figure_dpi": int(args.figure_dpi),
        "residue_label_shift": int(args.residue_label_shift),
        "calculation_repeated": False,
        "source_run_config": config,
    }
    (outdir / "redo_plot_config.json").write_text(
        json.dumps(redo_config, indent=2), encoding="utf-8"
    )
    print(f"Plot-only regeneration completed: {outdir.resolve()}")
    print("Accessibility calculations repeated: no")
    print(f"Primary figure: {(plots_dir / 'primary_summary.png').resolve()}")


def main() -> None:
    args = parse_args()
    if args.redo_plots is not None:
        redo_plots_from_existing_output(args)
        return

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, outdir)

    apo_files = sorted(str(p) for p in Path(args.apo_dir).glob(args.apo_glob) if p.is_file())
    bound_files = sorted(str(p) for p in Path(args.bound_dir).glob(args.bound_glob) if p.is_file())
    if not apo_files:
        raise SystemExit(f"No APO PDBs found in {args.apo_dir} with glob {args.apo_glob}")
    if not bound_files:
        raise SystemExit(f"No bound PDBs found in {args.bound_dir} with glob {args.bound_glob}")

    apo_weights = load_weight_map(args.apo_weights_csv, apo_files)
    bound_weights = load_weight_map(args.bound_weights_csv, bound_files)
    local_directions = fibonacci_sphere(args.n_directions)

    common = {
        "local_directions": local_directions,
        "target_residues": args.target_residues,
        "probe_radii": args.probe_radii,
        "approach_lengths": args.approach_lengths,
        "exclusion_windows": args.exclude_sequence_windows,
        "primary_window": args.primary_exclude_window,
        "primary_probe": args.primary_probe_radius,
        "primary_length": args.primary_approach_length,
        "convergence_counts": args.direction_convergence_counts,
        "ray_start": args.ray_start,
        "target_mode": args.target_mode,
        "gly_target_mode": args.gly_target_mode,
        "include_hetero": args.include_hetero,
        "bound_partner_chains": args.bound_partner_chains,
    }
    tasks: List[Dict[str, Any]] = []
    for path in apo_files:
        tasks.append({
            **common, "path": path, "ensemble_type": "APO",
            "prot_chain": args.apo_target_chain,
            "ensemble_weight": apo_weights[Path(path).name],
        })
    for path in bound_files:
        tasks.append({
            **common, "path": path, "ensemble_type": "BOUND",
            "prot_chain": args.bound_target_chain,
            "ensemble_weight": bound_weights[Path(path).name],
        })

    all_rows: List[Dict[str, Any]] = []
    all_convergence_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    map_weighted_sums: Dict[Tuple[str, int], np.ndarray] = {
        (condition, int(site)): np.zeros(args.n_directions, dtype=np.float64)
        for condition in CONDITION_ORDER for site in args.target_residues
    }
    map_weight_sums: Dict[Tuple[str, int], float] = {key: 0.0 for key in map_weighted_sums}

    with ProcessPoolExecutor(max_workers=max(1, args.n_workers)) as executor:
        future_to_task = {executor.submit(process_pdb_task, task): task for task in tasks}
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Analyzing conformers", unit="PDB"):
            task = future_to_task[future]
            try:
                result = future.result()
                all_rows.extend(result["rows"])
                all_convergence_rows.extend(result["convergence_rows"])
                weight = float(task["ensemble_weight"])
                for key_text, packed in result["maps"].items():
                    condition, site_text = key_text.split("|")
                    key = (condition, int(site_text))
                    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), count=args.n_directions).astype(np.float64)
                    map_weighted_sums[key] += weight * unpacked
                    map_weight_sums[key] += weight
            except Exception as exc:
                failures.append({
                    "source_pdb": os.path.abspath(str(task["path"])),
                    "ensemble_type": task["ensemble_type"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                if args.fail_fast:
                    raise
    print("Compiling results... this may take a moment for large ensembles...")
    pd.DataFrame(failures).to_csv(outdir / "failures.csv", index=False)
    if not all_rows:
        raise SystemExit("No conformers were analyzed successfully. See failures.csv.")

    per_df = pd.DataFrame(all_rows).sort_values([
        "condition", "source_name", "target_residue", "exclude_sequence_window",
        "probe_radius_A", "approach_length_A",
    ])
    per_df.to_csv(outdir / "per_conformer_accessibility.csv", index=False)
    residue_one_letters = build_residue_one_letter_map(per_df)

    region_per_df: Optional[pd.DataFrame] = None
    if args.region_residues is not None and args.region_name is not None:
        region_per_df = build_region_per_conformer(
            per_df, args.region_name, args.region_residues
        )
        region_per_df.to_csv(outdir / "region_per_conformer_accessibility.csv", index=False)

    condition_summary = summarize_conditions(per_df, args.bootstrap, args.seed)
    condition_summary.to_csv(outdir / "condition_summary.csv", index=False)

    effects = effect_decomposition(
        per_df, args.bootstrap, args.seed,
        args.primary_probe_radius, args.primary_approach_length,
        args.primary_exclude_window, args.minimum_meaningful_delta,
    )
    effects.to_csv(outdir / "accessibility_effect_decomposition.csv", index=False)

    primary_summary = plot_summary(
        condition_summary, effects, plots_dir,
        args.primary_probe_radius, args.primary_approach_length,
        args.primary_exclude_window, args.minimum_meaningful_delta,
        args.figure_dpi, residue_one_letters, args.residue_label_shift,
    )
    primary_summary.to_csv(outdir / "primary_summary.csv", index=False)

    if region_per_df is not None:
        region_summary = summarize_region_conditions(
            region_per_df, args.bootstrap, args.seed
        )
        region_summary.to_csv(outdir / "region_condition_summary.csv", index=False)
        region_effects = region_effect_decomposition(
            region_per_df, args.bootstrap, args.seed,
            args.primary_probe_radius, args.primary_approach_length,
            args.primary_exclude_window, args.minimum_meaningful_delta,
        )
        region_effects.to_csv(outdir / "region_effect_decomposition.csv", index=False)
        region_primary = plot_region_primary_summary(
            region_summary, region_effects, plots_dir,
            args.primary_probe_radius, args.primary_approach_length,
            args.primary_exclude_window, args.figure_dpi,
        )
        region_primary.to_csv(outdir / "region_primary_summary.csv", index=False)

    convergence_df = pd.DataFrame(all_convergence_rows)
    convergence_df.to_csv(outdir / "per_conformer_direction_convergence.csv", index=False)
    convergence_summary = summarize_direction_convergence(convergence_df, args.n_directions)
    convergence_summary.to_csv(outdir / "direction_convergence_summary.csv", index=False)
    plot_direction_convergence(convergence_summary, plots_dir, args.figure_dpi)

    exclusion_summary = plot_exclusion_window_sensitivity(
        condition_summary, plots_dir,
        args.primary_probe_radius, args.primary_approach_length,
        args.figure_dpi, residue_one_letters, args.residue_label_shift,
    )
    exclusion_summary.to_csv(outdir / "exclusion_window_sensitivity.csv", index=False)

    map_means: Dict[Tuple[str, int], np.ndarray] = {}
    for key, values in map_weighted_sums.items():
        denom = map_weight_sums[key]
        map_means[key] = (
            (values / denom).astype(np.float32)
            if denom > 0 else np.full(args.n_directions, np.nan, dtype=np.float32)
        )

    if args.save_directional_maps:
        payload: Dict[str, np.ndarray] = {
            "local_directions_xyz": local_directions.astype(np.float32),
            "primary_probe_radius_A": np.asarray(args.primary_probe_radius, dtype=np.float32),
            "primary_approach_length_A": np.asarray(args.primary_approach_length, dtype=np.float32),
            "primary_exclude_sequence_window": np.asarray(args.primary_exclude_window, dtype=np.int32),
        }
        for (condition, site), values in map_means.items():
            payload[f"{condition}_site_{site}"] = values
        np.savez_compressed(outdir / "directional_accessibility_maps_local_frame.npz", **payload)
        plot_directional_maps(
            map_means, local_directions, args.target_residues, plots_dir,
            args.primary_probe_radius, args.primary_approach_length,
            args.primary_exclude_window, args.figure_dpi,
            residue_one_letters, args.residue_label_shift,
        )

    readable = {
        "interpretation": "Probe-dependent accessible solid-angle fraction in a residue-local frame.",
        "primary_parameters": {
            "probe_radius_A": args.primary_probe_radius,
            "approach_length_A": args.primary_approach_length,
            "exclude_sequence_window": args.primary_exclude_window,
            "n_directions": args.n_directions,
        },
        "effect_sign": "Negative deltas indicate reduced accessibility.",
        "optional_region": (
            {
                "name": args.region_name,
                "residues": args.region_residues,
                "primary_endpoint": "within-conformer mean of residue accessible fractions",
                "secondary_endpoint": "within-conformer minimum residue accessible fraction",
            }
            if args.region_residues is not None else None
        ),
        "n_apo_input_pdbs": len(apo_files),
        "n_bound_input_pdbs": len(bound_files),
        "n_failed_pdbs": len(failures),
        "target_mode": args.target_mode,
        "gly_target_mode": args.gly_target_mode,
        "directional_frame": {
            "ser_thr_oxygen": "z=CB-to-Ogamma; x=projected CA-to-CB; y=right-handed cross-product",
            "sidechain_centroid": "z=CA-to-sidechain-centroid; x=projected CA-to-carbonyl-C; y=right-handed cross-product",
        },
        "direct_partner_inference": "Descriptive paired magnitude and prevalence; no direct-partner p-value is emphasized.",
        "outputs_are_png_only": True,
        "residue_plot_label_shift": args.residue_label_shift,
    }
    (outdir / "readable_summary.json").write_text(json.dumps(readable, indent=2), encoding="utf-8")

    print(f"Completed. Results written to: {outdir.resolve()}")
    print(f"Successful result rows: {len(per_df):,}")
    print(f"Failed PDBs: {len(failures)}")
    print(f"Figure: {(plots_dir / 'primary_summary.png').resolve()}")


if __name__ == "__main__":
    main()
