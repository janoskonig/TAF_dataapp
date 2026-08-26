"""
TAF Addon – Blender Addon
=========================
Fogászati morfometriai mérések, automatikus gerincél/áthajlás detektor
és betanítható landmark regresszió egy fájlban.

Modulok
-------
• TAF Morphometria  (N-panel → TAF → TAF Morphometria)
• Görbe Detektor    (N-panel → TAF → Görbe Detektor)
• Görbe Tanuló      (N-panel → TAF → Görbe Tanuló)

Telepítés
---------
  Edit → Preferences → Add-ons → Install → válaszd ezt a fájlt → engedélyezd

Előfeltétel: a fogászati modell okkluzális iránya = +Z (felfelé).

Szerzői jog: János König, 2024–2025
"""

bl_info = {
    "name": "TAF Addon",
    "author": "János König",
    "version": (2, 2, 0),
    "blender": (3, 0, 0),
    "location": "View3D › Sidebar › TAF",
    "description": "Morfometriai mérések (profil-export DB-be), gerincél detektor és landmark regresszió",
    "category": "Mesh",
}

import bpy
import bmesh
import csv
import json
import math
import os
from mathutils import Vector
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty,
    IntProperty, StringProperty,
)
from bpy.app.handlers import persistent
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup


ADDON_ID = __package__ or __name__


def _addon_preferences(context):
    """Return machine-level settings without storing secrets in patient files."""
    addon = context.preferences.addons.get(ADDON_ID)
    return addon.preferences if addon else None


@persistent
def _purge_legacy_scene_credentials(_unused=None):
    """Remove credentials saved by addon versions older than 2.2."""
    for scene in bpy.data.scenes:
        stored = scene.get("taf_props")
        if stored is None:
            continue
        for key in ("server_url", "api_key"):
            try:
                if key in stored:
                    del stored[key]
            except (KeyError, TypeError):
                pass


# ---------------------------------------------------------------------------
# Constants (shared by Curve Detector and Curve Learner)
# ---------------------------------------------------------------------------

CREST_NAME   = "TAF_Gerincelvonal"
BUCCAL_NAME  = "TAF_Bukkalis_athajlas"
LINGUAL_NAME = "TAF_Lingualis_athajlas"
LANDMARKS    = ("crest", "buccal", "lingual")
LM_CURVE     = {"crest": CREST_NAME, "buccal": BUCCAL_NAME, "lingual": LINGUAL_NAME}
LM_COLORS    = {
    "crest":   ("TAF_Gerincelvonal_Mat", (0.0, 0.25, 1.0, 1.0)),
    "buccal":  ("TAF_Athajlas_Mat",      (1.0, 0.4,  0.0, 1.0)),
    "lingual": ("TAF_Lingualis_Mat",     (0.0, 0.8,  0.2, 1.0)),
}

_CREST_TOL_MM = 0.4  # cluster tolerance around crest peak (mm)

# F6 – interalveolar angle: per-side segment between upper & lower model
# posterior ridge crest points, angle to occlusal plane (Z=0), sides averaged.
_F6_BAL_FELSO_NAME  = "TAF_F6_Bal_Felso"
_F6_BAL_ALSO_NAME   = "TAF_F6_Bal_Also"
_F6_JOBB_FELSO_NAME = "TAF_F6_Jobb_Felso"
_F6_JOBB_ALSO_NAME  = "TAF_F6_Jobb_Also"
# A10 – jaw relation: segment between upper & lower frontal ridge crest points
# (midsagittal plane), angle to the vertical axis.
_A10_FELSO_NAME = "TAF_A10_Felso"  # upper frontal ridge crest marker (midsagittal)
_A10_ALSO_NAME  = "TAF_A10_Also"   # lower frontal ridge crest marker (midsagittal)


# ===========================================================================
# Unit helpers
# ===========================================================================

def internal_to_mm(d_internal, scene_scale):
    if scene_scale <= 0:
        return d_internal
    return d_internal * scene_scale * 1000.0


def mm_to_internal(d_mm, scene_scale):
    if scene_scale <= 0:
        return d_mm
    return d_mm / (scene_scale * 1000.0)


# ===========================================================================
# Shared mesh helpers
# ===========================================================================

def _world_vertices(obj):
    """World-space vertex coordinates of the evaluated (modifier-applied) mesh."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = obj.evaluated_get(depsgraph)
    mesh      = eval_obj.to_mesh()
    mat       = obj.matrix_world
    verts     = [mat @ v.co.copy() for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    return verts


def _place_marker_at_selection(context, marker_name):
    """Place a sphere Empty at the centroid of selected vertices (Edit Mode).

    Returns (Vector position, None) on success, or (None, error_str) on failure.
    """
    obj = context.active_object
    if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
        return None, "Edit módban kell lenni egy mesh-szel!"
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    mat = obj.matrix_world
    sel = [mat @ v.co.copy() for v in bm.verts if v.select]
    if not sel:
        return None, "Nincs kijelölt vertex!"
    centroid = sum(sel, Vector()) / len(sel)
    scale = context.scene.unit_settings.scale_length or 1.0
    if marker_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[marker_name], do_unlink=True)
    marker = bpy.data.objects.new(marker_name, None)
    marker.empty_display_type = 'SPHERE'
    marker.empty_display_size = mm_to_internal(2.0, scale)
    marker.location = centroid
    context.collection.objects.link(marker)
    return centroid, None


def mesh_volume_mm3(obj, depsgraph):
    """Return mesh volume in mm³ for the given object (world space)."""
    bm = bmesh.new()
    eval_obj = obj.evaluated_get(depsgraph)
    bm.from_mesh(eval_obj.to_mesh())
    bm.transform(obj.matrix_world)
    volume = abs(bm.calc_volume(signed=True))
    bm.free()
    scale = bpy.context.scene.unit_settings.scale_length or 1.0
    # calc_volume() returns Blender-units³. BU³ → m³: ×scale³; m³ → mm³: ×1e9
    return volume * (scale ** 3) * 1e9


def _strip_extruded_plane(pts):
    """Traced anatomical lines are exported as thin *ribbon* meshes: the real
    edge (whose Z follows the anatomy) plus a flat reference plane created by
    extruding the curve to a constant height. That flat plane shows up as a
    coplanar, zero-variance row of vertices sitting at a global Z extreme — it
    must be dropped before measuring, exactly as the manual analysis does with
    its ``Z <= 30`` filter.

    `pts` is an (N, 3) numpy array (world coords, any unit). Returns the array
    with the flat reference plane removed. Parameter-free and scale-invariant
    (tolerances are relative to the cloud's Z-range); a no-op when no such plane
    is found, so genuine single-edge polylines pass through untouched."""
    import numpy as np
    if pts.shape[0] < 6:
        return pts
    z = pts[:, 2]
    zr = z.max() - z.min()
    if zr <= 0:
        return pts
    flat_tol = 0.02 * zr          # "flat" = Z spread within 2 % of the range
    best = None
    for extreme in (z.max(), z.min()):
        mask = np.abs(z - extreme) <= flat_tol
        # a real extruded plane: several coplanar verts at an extreme, clearly
        # separated (>20 % of the Z-range) from the rest of the cloud
        if 3 <= mask.sum() < len(z) and z[mask].std() <= flat_tol:
            gap = abs(z[~mask].mean() - extreme)
            if gap > 0.2 * zr and (best is None or mask.sum() > best.sum()):
                best = mask
    return pts[~best] if best is not None else pts


def avg_vertical_distance(obj_a, obj_b):
    """Vertical ridge-to-reference profile, mirroring the manual F1/A2 analysis.

    obj_a = ridge line, obj_b = reference line (buccal/lingual fold). Both are
    ribbon meshes, so the extruded flat reference plane is stripped first. The
    reference is then treated as a height function Z(X) and sampled at each ridge
    X via 1-D interpolation (``np.interp``, like the manual script) — only over
    the X-range the two lines share, to avoid extrapolating past the data.

    Returns (mean |Z_ridge - Z_ref| in mm, count, list of per-point dicts).
    distance_mm is **signed** (Z_ridge - Z_ref) so profile plots keep the sign;
    the scalar mean uses absolute values to match the manual ``mean(abs(...))``."""
    import numpy as np

    def world_pts(obj):
        return np.array([(obj.matrix_world @ v.co)[:] for v in obj.data.vertices],
                        dtype=float)

    pa = world_pts(obj_a)
    pb = world_pts(obj_b)
    if pa.size == 0 or pb.size == 0:
        return None, 0, []

    # Drop the flat extruded reference plane from each ribbon.
    pa = _strip_extruded_plane(pa)
    pb = _strip_extruded_plane(pb)
    if pa.size == 0 or pb.size == 0:
        return None, 0, []

    # Reference as a height function Z(X): sort by X, average duplicate X so
    # np.interp gets a strictly increasing, single-valued grid.
    order = np.argsort(pb[:, 0])
    bx_sorted, bz_sorted = pb[order, 0], pb[order, 2]
    ux, inv = np.unique(bx_sorted, return_inverse=True)
    uz = np.zeros_like(ux)
    np.add.at(uz, inv, bz_sorted)
    uz /= np.bincount(inv)

    # Only measure where the ridge overlaps the reference's X-range.
    lo, hi = ux.min(), ux.max()
    ridge = pa[(pa[:, 0] >= lo) & (pa[:, 0] <= hi)]
    ridge = ridge[np.argsort(ridge[:, 0])]
    if ridge.shape[0] == 0:
        return None, 0, []

    z_ref = np.interp(ridge[:, 0], ux, uz)
    signed = ridge[:, 2] - z_ref          # manual convention: signed height

    scale = bpy.context.scene.unit_settings.scale_length or 1.0
    points = []
    for (x, y, z), zr, sd in zip(ridge, z_ref, signed):
        points.append({
            "x_mm":        internal_to_mm(x,  scale),
            "y_mm":        internal_to_mm(y,  scale),
            "z_ridge_mm":  internal_to_mm(z,  scale),
            "z_ref_mm":    internal_to_mm(zr, scale),
            "distance_mm": internal_to_mm(sd, scale),   # signed
        })

    avg_mm = internal_to_mm(float(np.mean(np.abs(signed))), scale)
    return avg_mm, len(points), points


# ── A2 "Method 2": cross-sectional orthogonal standardized height ──────────────
# Port of the validated A2_comparison.py / A2_ribbon.py analysis. For each station
# along the ridge it measures the signed perpendicular distance from the ridge
# crest to the line joining the buccal and lingual functional sulci in the ridge's
# normal plane — an anatomically standardized ridge height, robust to the noisy
# vertical (Z-only) measurement used by the older A2 'b' method.

def _x_binned_median_curve(pts, n_bins=500):
    """Denoise an unordered vertex cloud into a representative curve by binning
    along X and taking the per-bin median (X, Y, Z)."""
    import numpy as np
    x = pts[:, 0]
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return pts
    edges = np.linspace(lo, hi, n_bins + 1)
    bid = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    rows = [np.median(pts[bid == b], axis=0) for b in np.unique(bid)]
    curve = np.asarray(rows)
    return curve[np.argsort(curve[:, 0])]


def _order_curve_nn(pts):
    """Chain unordered points into a continuous polyline by greedy
    nearest-neighbour, starting from the minimum-X point."""
    import numpy as np
    n = len(pts)
    if n < 3:
        return pts.copy()
    start = int(np.argmin(pts[:, 0]))
    remaining = set(range(n))
    remaining.discard(start)
    order = [start]
    cur = start
    while remaining:
        rem = np.array(list(remaining))
        nxt = int(rem[np.argmin(np.linalg.norm(pts[rem] - pts[cur], axis=1))])
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return pts[order]


def _resample_by_arclength(curve, n_samples=300):
    """Resample a 3D polyline to uniform arc-length stations."""
    import numpy as np
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total <= 0:
        return curve.copy()
    s = np.linspace(0.0, total, n_samples)
    return np.column_stack([np.interp(s, arc, curve[:, i]) for i in range(3)])


def _curve_tangent(curve, i):
    """Central-difference unit tangent at curve index i."""
    import numpy as np
    if i == 0:
        v = curve[1] - curve[0]
    elif i == len(curve) - 1:
        v = curve[-1] - curve[-2]
    else:
        v = curve[i + 1] - curve[i - 1]
    nv = np.linalg.norm(v)
    return v / nv if nv else np.array([1.0, 0.0, 0.0])


def _closest_on_curve_in_plane(curve, point, tangent):
    """Point on the polyline lying in the plane through `point` normal to
    `tangent` (segment/plane intersection); falls back to the nearest point on
    the polyline if no segment crosses the plane."""
    import numpy as np
    best, best_d, eps = None, np.inf, 1e-9
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        ab = b - a
        den = np.dot(ab, tangent)
        if abs(den) < eps:
            continue
        t = np.dot(point - a, tangent) / den
        if 0.0 <= t <= 1.0:
            q = a + t * ab
            d = np.linalg.norm(q - point)
            if d < best_d:
                best_d, best = d, q
    if best is not None:
        return best
    for i in range(len(curve) - 1):
        a, b = curve[i], curve[i + 1]
        ab = b - a
        den = np.dot(ab, ab)
        q = a if den <= eps else a + np.clip(np.dot(point - a, ab) / den, 0.0, 1.0) * ab
        d = np.linalg.norm(q - point)
        if d < best_d:
            best_d, best = d, q
    return best


def _project_to_segment(point, a, b):
    """Orthogonal projection of `point` onto segment AB (clamped)."""
    import numpy as np
    ab = b - a
    den = np.dot(ab, ab)
    if den == 0:
        return a
    return a + np.clip(np.dot(point - a, ab) / den, 0.0, 1.0) * ab


def a2_orthogonal_profile(obj_ridge, obj_buccal, obj_lingual,
                          n_bins=500, n_samples=300):
    """Cross-sectional standardized A2 ridge height (Method 2).

    Strips each ribbon's flat reference plane, clips the buccal/lingual sulci to
    the ridge's X-span (parameter-free; only measure where the ridge exists),
    builds denoised arch curves, resamples the ridge by arc length and, in each
    ridge normal plane, projects the crest orthogonally onto the buccal–lingual
    baseline. Returns (mean signed height in mm, count, list of per-point dicts
    with x, z (ridge), zref (projection) and d (signed orthogonal height))."""
    import numpy as np

    scale = bpy.context.scene.unit_settings.scale_length or 1.0

    def world_mm(obj):
        out = []
        for v in obj.data.vertices:
            co = obj.matrix_world @ v.co
            out.append([internal_to_mm(co.x, scale),
                        internal_to_mm(co.y, scale),
                        internal_to_mm(co.z, scale)])
        return np.asarray(out, dtype=float)

    clouds = [_strip_extruded_plane(world_mm(o))
              for o in (obj_ridge, obj_buccal, obj_lingual)]
    if min(len(c) for c in clouds) < 3:
        return None, 0, []

    # Identify the ridge automatically (highest mean Z = crest, sitting above
    # both sulci) so the result does not depend on selection order. The two
    # sulci are interchangeable — the buccal–lingual baseline segment is the
    # same regardless of which endpoint is buccal vs lingual.
    ridge_i = max(range(3), key=lambda k: clouds[k][:, 2].mean())
    R = clouds[ridge_i]
    sulci = [clouds[k] for k in range(3) if k != ridge_i]
    B, L = sulci[0], sulci[1]

    # Only measure along the ridge's own X-extent.
    xlo, xhi = R[:, 0].min(), R[:, 0].max()
    clip = lambda p: p[(p[:, 0] >= xlo) & (p[:, 0] <= xhi)]
    B, L = clip(B), clip(L)
    if min(len(B), len(L)) < 3:
        return None, 0, []

    ridge_curve = _order_curve_nn(_x_binned_median_curve(R, n_bins))
    buccal_curve = _order_curve_nn(_x_binned_median_curve(B, n_bins))
    lingual_curve = _order_curve_nn(_x_binned_median_curve(L, n_bins))
    ridge = _resample_by_arclength(ridge_curve, n_samples)

    points, heights = [], []
    for i, rp in enumerate(ridge):
        tan = _curve_tangent(ridge, i)
        bp = _closest_on_curve_in_plane(buccal_curve, rp, tan)
        lp = _closest_on_curve_in_plane(lingual_curve, rp, tan)
        if bp is None or lp is None:
            continue
        proj = _project_to_segment(rp, lp, bp)
        dist = float(np.linalg.norm(rp - proj))
        sign = np.sign(rp[2] - proj[2])
        sd = dist if sign == 0 else float(sign) * dist
        heights.append(sd)
        points.append({
            "x":    round(float(rp[0]), 3),
            "z":    round(float(rp[2]), 3),
            "zref": round(float(proj[2]), 3),
            "d":    round(sd, 3),
        })

    if not heights:
        return None, 0, []
    return float(np.mean(heights)), len(points), points


# ── A2 "Ribbon method": ridge-to-fold-surface distance ────────────────────────
# The buccal + lingual sulci are joined in Blender into one strip surface, and the
# ridge crest into a vertical ribbon. The standardized height is the signed
# distance from each ridge-crest station to the closest point on the fold strip
# surface — a single mesh-to-surface projection, no per-curve correspondence.

def _closest_point_on_triangle(p, a, b, c):
    """Closest point on triangle ABC to point p (Ericson, Real-Time Collision
    Detection). All args are length-3 numpy arrays."""
    import numpy as np
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab @ ap
    d2 = ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3 = ab @ bp
    d4 = ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5 = ab @ cp
    d6 = ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + ab * (vb * denom) + ac * (vc * denom)


def a2_ribbon_profile(obj_a, obj_b, n_bins=500, n_samples=300):
    """Ribbon-based A2 standardized height (Method 'c').

    Two objects: a ridge-crest ribbon and the joined buccal+lingual fold strip
    surface (order-independent). Strips the ridge ribbon's flat extruded plane,
    resamples the crest by arc length, and for each crest station takes the
    signed distance to the closest point on the fold strip surface (positive =
    crest above the surface in Z). Returns (mean signed height in mm, count,
    per-point dicts with x, z (crest), zref (surface) and d (signed))."""
    import numpy as np

    scale = bpy.context.scene.unit_settings.scale_length or 1.0

    def world_mm_verts(obj):
        return np.asarray(
            [[internal_to_mm(c, scale) for c in (obj.matrix_world @ v.co)[:3]]
             for v in obj.data.vertices], dtype=float)

    def world_mm_tris(obj):
        vw = [np.array([internal_to_mm(c, scale) for c in (obj.matrix_world @ v.co)[:3]],
                       dtype=float) for v in obj.data.vertices]
        tris = []
        for poly in obj.data.polygons:
            idx = list(poly.vertices)
            for k in range(1, len(idx) - 1):     # fan-triangulate n-gons
                tris.append((vw[idx[0]], vw[idx[k]], vw[idx[k + 1]]))
        return tris

    va = _strip_extruded_plane(world_mm_verts(obj_a))
    vb = _strip_extruded_plane(world_mm_verts(obj_b))
    if min(len(va), len(vb)) < 3:
        return None, 0, []

    # The ridge crest sits above the fold strip → higher mean Z identifies it.
    if va[:, 2].mean() >= vb[:, 2].mean():
        ridge_pts, surf_obj = va, obj_b
    else:
        ridge_pts, surf_obj = vb, obj_a

    crest = _resample_by_arclength(
        _order_curve_nn(_x_binned_median_curve(ridge_pts, n_bins)), n_samples)
    tris = world_mm_tris(surf_obj)
    if not tris:
        return None, 0, []

    points, heights = [], []
    for rp in crest:
        best_q, best_d = None, float('inf')
        for (a, b, c) in tris:
            q = _closest_point_on_triangle(rp, a, b, c)
            d = float(np.linalg.norm(q - rp))
            if d < best_d:
                best_d, best_q = d, q
        sign = np.sign(rp[2] - best_q[2])
        sd = best_d if sign == 0 else float(sign) * best_d
        heights.append(sd)
        points.append({
            "x":    round(float(rp[0]), 3),
            "z":    round(float(rp[2]), 3),
            "zref": round(float(best_q[2]), 3),
            "d":    round(sd, 3),
        })

    if not heights:
        return None, 0, []
    return float(np.mean(heights)), len(points), points


def ridge_arclength_mm(obj, bin_mm=2.0):
    """True 3D arc length (mm) of a ridge curve, robust to trace noise.

    Strips the ribbon's flat reference plane, denoises with fixed-width
    (`bin_mm`) X-binning + per-bin median, then sums 3D segment lengths along X.
    Fixed bin width keeps the result stable regardless of point density (raw
    nearest-neighbour chaining over-estimates badly on noisy traces). Used to
    size-standardize the F2 undercut volume (V / L³). Returns 0.0 if too short."""
    import numpy as np
    scale = bpy.context.scene.unit_settings.scale_length or 1.0
    pts = np.asarray(
        [[internal_to_mm(c, scale) for c in (obj.matrix_world @ v.co)[:3]]
         for v in obj.data.vertices], dtype=float)
    pts = _strip_extruded_plane(pts)
    if len(pts) < 2:
        return 0.0
    x = pts[:, 0]
    ext = x.max() - x.min()
    if ext <= 0:
        return 0.0
    nb = max(4, int(round(ext / bin_mm)))
    edges = np.linspace(x.min(), x.max(), nb + 1)
    bid = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    rows = [np.median(pts[bid == b], axis=0) for b in np.unique(bid)]
    curve = np.asarray(rows)
    curve = curve[np.argsort(curve[:, 0])]
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def _write_raw_distances_csv(csv_path, patient_id, measurement, points):
    """Append per-point raw data for a measurement to a CSV next to csv_path.

    `points` is a list of dicts from avg_vertical_distance() with x_mm, y_mm,
    z_ridge_mm, z_ref_mm, distance_mm — enough to rebuild profile plots
    (X position along arch vs ridge height) and compute CI/PI bands.
    """
    if not csv_path or not points:
        return None
    dirpath = os.path.dirname(bpy.path.abspath(csv_path))
    safe_id  = patient_id.strip().replace(" ", "_").replace("/", "-")
    filepath = os.path.join(dirpath, f"{safe_id}_{measurement}_raw.csv")
    file_exists = os.path.isfile(filepath)
    fields = ["patient_id", "measurement", "index",
              "x_mm", "y_mm", "z_ridge_mm", "z_ref_mm", "distance_mm"]
    with open(filepath, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for i, p in enumerate(points):
            writer.writerow({
                "patient_id":  patient_id.strip(),
                "measurement": measurement,
                "index":       i,
                "x_mm":        round(p["x_mm"], 4),
                "y_mm":        round(p["y_mm"], 4),
                "z_ridge_mm":  round(p["z_ridge_mm"], 4),
                "z_ref_mm":    round(p["z_ref_mm"], 4),
                "distance_mm": round(p["distance_mm"], 4),
            })
    return filepath


def _profile_json(points, side=None):
    """Compact JSON string of profile points for DB storage.

    Keeps x_mm, z_ridge_mm, z_ref_mm, distance_mm (rounded to 3 decimals);
    an optional `side` tag ('buccal'/'lingual') distinguishes A2 method-b curves.
    Returns "" for empty input.
    """
    if not points:
        return ""
    out = []
    for p in points:
        rec = {
            "x":    round(p["x_mm"], 3),
            "z":    round(p["z_ridge_mm"], 3),
            "zref": round(p["z_ref_mm"], 3),
            "d":    round(p["distance_mm"], 3),
        }
        if side:
            rec["side"] = side
        out.append(rec)
    return json.dumps(out, ensure_ascii=False)


# ===========================================================================
# Arch slicing infrastructure  (shared by Curve Detector and Curve Learner)
# ===========================================================================

class SliceFrame:
    """One cross-section slicing frame."""
    __slots__ = ("co", "normal", "lateral")
    def __init__(self, co, normal, lateral):
        self.co = co; self.normal = normal; self.lateral = lateral


def _bbox_world(obj):
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    return (
        [v.x for v in corners],
        [v.y for v in corners],
        [v.z for v in corners],
    )


def _median(vals):
    s = sorted(vals)
    k = len(s)
    if k == 0:
        return 0.0
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


def _fit_arch_midline_xy(obj, props):
    """Fit an ordered arch-midline polyline in the XY plane. Returns (midline_pts, pole)."""
    verts = _world_vertices(obj)
    if len(verts) < 8:
        return None, None

    zs = [v.z for v in verts]
    zmin, zmax = min(zs), max(zs)
    zmid = 0.5 * (zmin + zmax)

    region = verts
    if props.use_ridge_region and (zmax - zmin) > 1e-9:
        z_thr  = zmax - (zmax - zmin) * (props.ridge_region_pct / 100.0)
        region = [v for v in verts if v.z >= z_thr]
        if len(region) < 8:
            region = verts

    px = sum(v.x for v in region) / len(region)
    py = sum(v.y for v in region) / len(region)
    pole = Vector((px, py, 0.0))

    ang = []
    for v in region:
        dx, dy = v.x - px, v.y - py
        if dx * dx + dy * dy >= 1e-12:
            ang.append((math.atan2(dy, dx), v))
    if len(ang) < 8:
        return None, None
    ang.sort(key=lambda t: t[0])

    angles = [a for a, _ in ang]
    m = len(angles)
    best_gap, best_i = -1.0, 0
    for i in range(m):
        a0 = angles[i]
        a1 = angles[(i + 1) % m] + (2 * math.pi if i + 1 == m else 0.0)
        if a1 - a0 > best_gap:
            best_gap, best_i = a1 - a0, i
    sweep_start = angles[(best_i + 1) % m]

    def rel(a):
        r = a - sweep_start
        while r < 0.0:
            r += 2 * math.pi
        return r

    swept   = sorted(((rel(a), v) for a, v in ang), key=lambda t: t[0])
    r_min   = swept[0][0]
    r_max   = swept[-1][0]
    span    = r_max - r_min
    if span < 1e-6:
        return None, None

    margin = props.margin_pct / 100.0
    lo = r_min + span * margin
    hi = r_max - span * margin
    if hi - lo < 1e-6:
        lo, hi = r_min, r_max

    n_bins = max(8, props.n_slices)
    bins   = [[] for _ in range(n_bins)]
    for rA, v in swept:
        if lo <= rA <= hi:
            idx = max(0, min(n_bins - 1, int((rA - lo) / (hi - lo) * (n_bins - 1) + 0.5)))
            bins[idx].append(v)

    midline = []
    for b in bins:
        if b:
            midline.append(Vector((_median([v.x for v in b]),
                                   _median([v.y for v in b]),
                                   zmid)))
    if len(midline) < 3:
        return None, None
    return midline, pole


def _resample_polyline(pts, n):
    """Resample an ordered polyline to n points evenly by arc length.
    Returns list of (point, tangent)."""
    if len(pts) < 2:
        return []
    lengths = [0.0]
    for i in range(1, len(pts)):
        lengths.append(lengths[-1] + (pts[i] - pts[i - 1]).length)
    total = lengths[-1]
    if total < 1e-8:
        return []

    out = []
    for i in range(n):
        s = total * i / (n - 1)
        idx = 0
        for j in range(len(lengths) - 1):
            if lengths[j] <= s <= lengths[j + 1]:
                idx = j
                break
        seg = lengths[idx + 1] - lengths[idx]
        if seg < 1e-8:
            co = pts[idx].copy()
        else:
            t  = (s - lengths[idx]) / seg
            co = pts[idx].lerp(pts[idx + 1], t)
        a   = pts[max(0, idx)]
        b   = pts[min(idx + 1, len(pts) - 1)]
        tan = (b - a)
        tan.z = 0.0
        if tan.length < 1e-8:
            tan = Vector((1.0, 0.0, 0.0))
        out.append((co, tan.normalized()))
    return out


def _frames_from_samples(samples, pole):
    """Build SliceFrame list from (point, tangent) samples."""
    frames = []
    for co, tan in samples:
        lateral = Vector((-tan.y, tan.x, 0.0))
        if lateral.length < 1e-8:
            lateral = Vector((0.0, 1.0, 0.0))
        lateral.normalize()
        outward = Vector((co.x - pole.x, co.y - pole.y, 0.0))
        if lateral.dot(outward) < 0.0:
            lateral = -lateral
        frames.append(SliceFrame(co.copy(), tan.copy(), lateral))
    return frames


def _frames_parallel(obj, props, axis):
    """X / Y parallel-plane fallback frames."""
    xs, ys, zs = _bbox_world(obj)
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))
    pole = Vector((cx, cy, 0.0))
    n      = props.n_slices
    margin = props.margin_pct / 100.0
    frames = []
    if axis == 'X':
        rng = max(xs) - min(xs)
        lo  = min(xs) + rng * margin
        hi  = max(xs) - rng * margin
        for i in range(n):
            x = lo + (hi - lo) * i / (n - 1)
            frames.append(SliceFrame(Vector((x, cy, cz)),
                                     Vector((1.0, 0.0, 0.0)),
                                     Vector((0.0, 1.0, 0.0))))
    else:
        rng = max(ys) - min(ys)
        lo  = min(ys) + rng * margin
        hi  = max(ys) - rng * margin
        for i in range(n):
            y = lo + (hi - lo) * i / (n - 1)
            frames.append(SliceFrame(Vector((cx, y, cz)),
                                     Vector((0.0, 1.0, 0.0)),
                                     Vector((1.0, 0.0, 0.0))))
    return frames, pole


def _frames_from_curve(curve_obj, props, pole):
    """Frames sampled along a manual Arch_Axis curve object."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = curve_obj.evaluated_get(depsgraph)
    mat       = curve_obj.matrix_world
    mesh      = eval_obj.to_mesh()
    verts     = [mat @ v.co.copy() for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    if len(verts) < 2:
        return []
    return _frames_from_samples(_resample_polyline(verts, props.n_slices), pole)


def get_slice_frames(obj, props):
    """Dispatch slicing strategy → (list[SliceFrame], pole)."""
    mode = props.arch_axis

    if mode == 'X':
        return _frames_parallel(obj, props, 'X')
    if mode == 'Y':
        return _frames_parallel(obj, props, 'Y')

    if mode == 'CURVE':
        arch_obj = bpy.data.objects.get("Arch_Axis")
        xs, ys, _ = _bbox_world(obj)
        pole = Vector((0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.0))
        if arch_obj and arch_obj.type == 'CURVE':
            frames = _frames_from_curve(arch_obj, props, pole)
            if frames:
                return frames, pole

    midline, pole = _fit_arch_midline_xy(obj, props)
    if midline is not None:
        frames = _frames_from_samples(_resample_polyline(midline, props.n_slices), pole)
        if frames:
            return frames, pole

    xs, ys, _ = _bbox_world(obj)
    axis = 'X' if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 'Y'
    return _frames_parallel(obj, props, axis)


# ===========================================================================
# Edge / point slicing
# ===========================================================================

def slice_mesh_edges(obj, plane_co, plane_no, reject_border=True):
    """Intersect every mesh edge with the plane.
    Returns [(point3d, is_border), ...]. Border intersections can be filtered."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = obj.evaluated_get(depsgraph)
    mesh      = eval_obj.to_mesh()
    mat       = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    n   = plane_no.normalized()
    out = []
    for edge in bm.edges:
        v0 = mat @ edge.verts[0].co
        v1 = mat @ edge.verts[1].co
        d0 = (v0 - plane_co).dot(n)
        d1 = (v1 - plane_co).dot(n)
        if d0 * d1 < 0.0:
            is_border = (len(edge.link_faces) < 2)
            if is_border and reject_border:
                continue
            t = d0 / (d0 - d1)
            out.append((v0.lerp(v1, t), is_border))
    bm.free()
    eval_obj.to_mesh_clear()
    return out


def slice_mesh_points(obj, plane_co, plane_no):
    """Intersect every non-border mesh edge with the plane. Returns [point3d, ...]."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = obj.evaluated_get(depsgraph)
    mesh      = eval_obj.to_mesh()
    mat       = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(mesh)
    n   = plane_no.normalized()
    out = []
    for edge in bm.edges:
        v0 = mat @ edge.verts[0].co
        v1 = mat @ edge.verts[1].co
        d0 = (v0 - plane_co).dot(n)
        d1 = (v1 - plane_co).dot(n)
        if d0 * d1 < 0.0:
            if len(edge.link_faces) < 2:
                continue
            t = d0 / (d0 - d1)
            out.append(v0.lerp(v1, t))
    bm.free()
    eval_obj.to_mesh_clear()
    return out


def evaluated_curve_segments(curve_obj):
    """World-space centerline segments of a curve (bevel suppressed, modifiers applied)."""
    data      = curve_obj.data
    old_bevel = getattr(data, "bevel_depth", 0.0)
    old_extr  = getattr(data, "extrude", 0.0)
    changed   = bool(old_bevel) or bool(old_extr)
    try:
        if changed:
            try: data.bevel_depth = 0.0
            except Exception: pass
            try: data.extrude = 0.0
            except Exception: pass
            bpy.context.view_layer.update()
        dg   = bpy.context.evaluated_depsgraph_get()
        eo   = curve_obj.evaluated_get(dg)
        mat  = curve_obj.matrix_world
        me   = eo.to_mesh()
        segs = []
        for e in me.edges:
            v0 = mat @ me.vertices[e.vertices[0]].co.copy()
            v1 = mat @ me.vertices[e.vertices[1]].co.copy()
            segs.append((v0, v1))
        eo.to_mesh_clear()
    finally:
        if changed:
            try: data.bevel_depth = old_bevel
            except Exception: pass
            try: data.extrude = old_extr
            except Exception: pass
            bpy.context.view_layer.update()
    return segs


def crossings_from_segments(segs, plane_co, plane_no):
    """Intersect precomputed (v0,v1) segments with a plane → list of points."""
    n   = plane_no.normalized()
    pts = []
    for v0, v1 in segs:
        d0 = (v0 - plane_co).dot(n)
        d1 = (v1 - plane_co).dot(n)
        if d0 * d1 < 0.0:
            t = d0 / (d0 - d1)
            pts.append(v0.lerp(v1, t))
    return pts


# ===========================================================================
# Profile projections
# ===========================================================================

def _to_profile(slice_pts, frame):
    """Project detector slice points (point, is_border) into local (L, H) frame.
    Returns list of dicts sorted by L."""
    prof = []
    for p, border in slice_pts:
        L = (p - frame.co).dot(frame.lateral)
        prof.append({'L': L, 'H': p.z, 'p': p, 'border': border})
    prof.sort(key=lambda d: d['L'])
    return prof


def _to_profile_ml(slice_pts, frame):
    """Project ML slice points (plain Vectors) into local (L, H) frame.
    Returns list of dicts sorted by L."""
    prof = [{'L': (p - frame.co).dot(frame.lateral), 'H': p.z, 'p': p}
            for p in slice_pts]
    prof.sort(key=lambda d: d['L'])
    return prof


# ===========================================================================
# Smoothing + curve creation  (shared)
# ===========================================================================

def _smooth_sequence(pts, strength):
    """MAD-based outlier rejection + moving-average smoothing."""
    n = len(pts)
    if strength <= 0 or n < 5:
        return [p.copy() for p in pts]

    coords = [[p.x for p in pts], [p.y for p in pts], [p.z for p in pts]]
    half, k_mad = 2, 3.5
    for axis in coords:
        flagged = set()
        for i in range(n):
            window  = axis[max(0, i - half):min(n, i + half + 1)]
            med     = _median(window)
            absdev  = [abs(x - med) for x in window]
            mad     = sorted(absdev)[len(absdev) // 2] if absdev else 0.0
            if mad > 1e-9:
                scale, thr = mad, k_mad * mad
            else:
                mean_ad = sum(absdev) / len(absdev) if absdev else 0.0
                scale, thr = mean_ad, 3.0 * mean_ad
            if scale > 1e-9 and abs(axis[i] - med) > thr:
                flagged.add(i)
        for i in flagged:
            lo, hi = i - 1, i + 1
            while lo >= 0 and lo in flagged:
                lo -= 1
            while hi < n and hi in flagged:
                hi += 1
            if lo >= 0 and hi < n:
                t = (i - lo) / (hi - lo)
                axis[i] = axis[lo] * (1 - t) + axis[hi] * t
            elif lo >= 0:
                axis[i] = axis[lo]
            elif hi < n:
                axis[i] = axis[hi]
    win = min(strength, (n - 1) // 2)
    if win >= 1:
        for axis in coords:
            src = axis[:]
            for i in range(1, n - 1):
                seg = src[max(0, i - win):min(n, i + win + 1)]
                axis[i] = sum(seg) / len(seg)
    return [Vector((coords[0][i], coords[1][i], coords[2][i])) for i in range(n)]


def make_nurbs_curve(name, pts_3d, resolution, smooth):
    """Create (or replace) a 3D NURBS curve object from a list of Vectors."""
    if name in bpy.data.objects:
        old_obj  = bpy.data.objects[name]
        old_data = old_obj.data
        bpy.data.objects.remove(old_obj, do_unlink=True)
        if old_data and old_data.users == 0 and isinstance(old_data, bpy.types.Curve):
            bpy.data.curves.remove(old_data)
    cd = bpy.data.curves.new(name, type='CURVE')
    cd.dimensions          = '3D'
    cd.resolution_u        = resolution
    cd.render_resolution_u = resolution * 2
    sp = cd.splines.new('NURBS')
    sp.points.add(len(pts_3d) - 1)
    for i, pt in enumerate(pts_3d):
        sp.points[i].co = (pt.x, pt.y, pt.z, 1.0)
    sp.use_endpoint_u = True
    sp.order_u = 4 if (smooth and len(pts_3d) >= 4) else 3
    curve_obj = bpy.data.objects.new(name, cd)
    bpy.context.collection.objects.link(curve_obj)
    return curve_obj


def _assign_material(curve_obj, mat_name, rgba):
    if not curve_obj.data.materials:
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
        mat.diffuse_color = rgba
        curve_obj.data.materials.append(mat)


# ===========================================================================
# Curve Detector – landmark detection
# ===========================================================================

def _detect_crest(prof, crest_win_internal, tol_internal):
    central = [d for d in prof if abs(d['L']) <= crest_win_internal] or prof
    if not central:
        return None
    h_max   = max(d['H'] for d in central)
    cluster = [d for d in central if d['H'] >= h_max - tol_internal]
    return min(cluster, key=lambda d: abs(d['L']))


def _detect_fold(prof, crest_L, side, riseback_internal):
    if side > 0:
        pts = sorted([d for d in prof if d['L'] > crest_L + 1e-6], key=lambda d:  d['L'])
    else:
        pts = sorted([d for d in prof if d['L'] < crest_L - 1e-6], key=lambda d: -d['L'])
    if len(pts) < 3:
        return None, False

    n_bins = min(12, max(4, len(pts) // 3))
    L0     = pts[0]['L']
    L1     = pts[-1]['L']
    span   = abs(L1 - L0)
    if span < 1e-9:
        return None, False

    bins = [[] for _ in range(n_bins)]
    for d in pts:
        frac = abs(d['L'] - L0) / span
        idx  = max(0, min(n_bins - 1, int(frac * (n_bins - 1) + 0.5)))
        bins[idx].append(d)

    bin_h = []
    for bi, b in enumerate(bins):
        if not b:
            continue
        med_h = _median([d['H'] for d in b])
        rep   = min(b, key=lambda d: abs(d['H'] - med_h))
        bin_h.append((bi, med_h, rep))
    if len(bin_h) < 2:
        return None, False

    low_k    = min(range(len(bin_h)), key=lambda k: bin_h[k][1])
    low_h    = bin_h[low_k][1]
    fold_rec = bin_h[low_k][2]
    confident = any(bin_h[k][1] >= low_h + riseback_internal
                    for k in range(low_k + 1, len(bin_h)))
    return fold_rec, confident


def _point_to_segment_distance_3d(p, a, b):
    ab = b - a
    ab_len_sq = ab.dot(ab)
    if ab_len_sq < 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab_len_sq))
    return (p - (a + t * ab)).length


def detect_upper(slice_pts, frame, props, scene_scale):
    if len(slice_pts) < 4:
        return None, None
    prof  = _to_profile(slice_pts, frame)
    win   = mm_to_internal(props.crest_window_mm, scene_scale)
    tol   = mm_to_internal(_CREST_TOL_MM, scene_scale)
    crest = _detect_crest(prof, win, tol)
    if crest is None:
        return None, None
    rise = mm_to_internal(props.fold_riseback_mm, scene_scale)
    fold, _ = _detect_fold(prof, crest['L'], +1, rise)
    if fold is None:
        fold, _ = _detect_fold(prof, crest['L'], -1, rise)
    if fold is None:
        return crest['p'], None
    if internal_to_mm(crest['H'] - fold['H'], scene_scale) < props.min_sulcus_drop:
        return crest['p'], None
    return crest['p'], fold['p']


def detect_lower(slice_pts, frame, props, scene_scale):
    if len(slice_pts) < 5:
        return None, None, None, 0.0
    prof  = _to_profile(slice_pts, frame)
    win   = mm_to_internal(props.crest_window_mm, scene_scale)
    tol   = mm_to_internal(_CREST_TOL_MM, scene_scale)
    crest = _detect_crest(prof, win, tol)
    if crest is None:
        return None, None, None, 0.0
    rise = mm_to_internal(props.fold_riseback_mm, scene_scale)
    buc, _ = _detect_fold(prof, crest['L'], +1, rise)
    lin, _ = _detect_fold(prof, crest['L'], -1, rise)
    if buc is None or lin is None:
        return None, None, None, 0.0
    ortho = _point_to_segment_distance_3d(crest['p'], lin['p'], buc['p'])
    if internal_to_mm(ortho, scene_scale) < props.min_sulcus_drop:
        return None, None, None, 0.0
    return crest['p'], buc['p'], lin['p'], ortho


# ===========================================================================
# Curve Learner – feature extraction
# ===========================================================================

def _fill_gaps(arr):
    n     = len(arr)
    known = [i for i in range(n) if arr[i] is not None]
    if not known:
        return [0.0] * n
    for i in range(n):
        if arr[i] is None:
            lo = max((k for k in known if k < i), default=None)
            hi = min((k for k in known if k > i), default=None)
            if lo is not None and hi is not None:
                t      = (i - lo) / (hi - lo)
                arr[i] = arr[lo] * (1 - t) + arr[hi] * t
            elif lo is not None:
                arr[i] = arr[lo]
            else:
                arr[i] = arr[hi]
    return arr


def build_features(prof, K, scene_scale):
    """Fixed-length feature vector for a slice profile. Returns (feat, half_width, Hmin, Hspan) or None."""
    if len(prof) < 4:
        return None
    Ls   = [d['L'] for d in prof]
    Hs   = [d['H'] for d in prof]
    Lmin, Lmax = min(Ls), max(Ls)
    Hmin, Hmax = min(Hs), max(Hs)
    half_width = max(abs(Lmin), abs(Lmax), 1e-6)
    Hspan      = max(Hmax - Hmin, 1e-6)
    binw       = (2.0 / (K - 1)) * half_width
    up = [None] * K
    lo = [None] * K
    for k in range(K):
        Lc  = (-1.0 + 2.0 * k / (K - 1)) * half_width
        sel = [d for d in prof if abs(d['L'] - Lc) <= binw]
        if sel:
            up[k] = max(d['H'] for d in sel)
            lo[k] = min(d['H'] for d in sel)
    up = _fill_gaps(up)
    lo = _fill_gaps(lo)
    up_n = [(h - Hmin) / Hspan for h in up]
    lo_n = [(h - Hmin) / Hspan for h in lo]
    feat = up_n + lo_n + [
        internal_to_mm(half_width, scene_scale),
        internal_to_mm(Hspan,      scene_scale),
        internal_to_mm(Lmin,       scene_scale),
    ]
    return feat, half_width, Hmin, Hspan


def _target_L(crossings, frame):
    if not crossings:
        return None
    Ls = [(p - frame.co).dot(frame.lateral) for p in crossings]
    return _median(Ls)


def extract_samples(obj, props, scene_scale, cast_id, jaw):
    frames, _ = get_slice_frames(obj, props)
    curve_segs = {}
    for lm, name in LM_CURVE.items():
        c = bpy.data.objects.get(name)
        if c is not None and c.type == 'CURVE':
            curve_segs[lm] = evaluated_curve_segments(c)
    samples = []
    for fr in frames:
        pts      = slice_mesh_points(obj, fr.co, fr.normal)
        feat_pack = build_features(_to_profile_ml(pts, fr), props.feat_samples, scene_scale)
        if feat_pack is None:
            continue
        feat, half_width, _, _ = feat_pack
        rec = {"cast": cast_id, "jaw": jaw, "feat": feat,
               "half_width_mm": internal_to_mm(half_width, scene_scale)}
        any_t = False
        for lm in LANDMARKS:
            segs = curve_segs.get(lm)
            t    = None
            if segs:
                L = _target_L(crossings_from_segments(segs, fr.co, fr.normal), fr)
                if L is not None:
                    t     = L / half_width
                    any_t = True
            rec["t_" + lm] = t
        if any_t:
            samples.append(rec)
    return samples


# ===========================================================================
# Curve Learner – dataset / model JSON IO
# ===========================================================================

def _abspath(p):
    return bpy.path.abspath(p) if p else p


def load_dataset(path):
    p = _abspath(path)
    if p and os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"feat_dim": None, "K": None, "samples": []}


def save_dataset(path, data):
    p = _abspath(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return p


# ===========================================================================
# Curve Learner – ridge regression
# ===========================================================================

def ridge_fit(X, y, lam):
    import numpy as np
    X  = np.asarray(X, dtype=float)
    y  = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    n, d = Xs.shape
    Xa = np.hstack([Xs, np.ones((n, 1))])
    A  = Xa.T @ Xa
    reg = np.eye(d + 1) * lam
    reg[d, d] = 0.0
    w = np.linalg.solve(A + reg, Xa.T @ y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def ridge_predict(model, x):
    import numpy as np
    mu = np.asarray(model["mu"])
    sd = np.asarray(model["sd"])
    w  = np.asarray(model["w"])
    xs = (np.asarray(x, dtype=float) - mu) / sd
    return float(np.append(xs, 1.0) @ w)


def train_all(dataset, lam):
    models = {}
    counts = {}
    for jaw in ("UPPER", "LOWER"):
        for lm in LANDMARKS:
            X, y = [], []
            for s in dataset["samples"]:
                if s.get("jaw") != jaw:
                    continue
                t = s.get("t_" + lm)
                if t is None:
                    continue
                X.append(s["feat"]); y.append(t)
            if len(X) >= 8:
                models.setdefault(jaw, {})[lm] = ridge_fit(X, y, lam)
                counts[(jaw, lm)] = len(X)
    return models, counts


def loo_eval(dataset, lam):
    samples = dataset["samples"]
    casts   = sorted({s["cast"] for s in samples})
    results = {}
    for jaw in ("UPPER", "LOWER"):
        for lm in LANDMARKS:
            errs = []
            for held in casts:
                tr_X, tr_y, te = [], [], []
                for s in samples:
                    if s.get("jaw") != jaw or s.get("t_" + lm) is None:
                        continue
                    if s["cast"] == held:
                        te.append(s)
                    else:
                        tr_X.append(s["feat"]); tr_y.append(s["t_" + lm])
                if len(tr_X) < 8 or not te:
                    continue
                mdl = ridge_fit(tr_X, tr_y, lam)
                for s in te:
                    errs.append(abs(ridge_predict(mdl, s["feat"]) - s["t_" + lm])
                                * s["half_width_mm"])
            if errs:
                results[(jaw, lm)] = (sum(errs) / len(errs), len(errs))
    return results


def _snap_landmark(prof, L_pred, lm, snap_internal):
    win = snap_internal
    for _ in range(4):
        cand = [d for d in prof if abs(d['L'] - L_pred) <= win]
        if cand:
            if lm == "crest":
                return max(cand, key=lambda d: d['H'])['p']
            return min(cand, key=lambda d: d['H'])['p']
        win *= 2.0
    return min(prof, key=lambda d: abs(d['L'] - L_pred))['p']


# ===========================================================================
# ── MORPHOMETRIA ─────────────────────────────────────────────────────────────
# ===========================================================================

class TAF_Props(PropertyGroup):

    patient_id: StringProperty(
        name="TAJ szám",
        description="Páciens TAJ-száma (pl. 000-111-222). Automatikusan kitöltődik a fájlnévből.",
        default=""
    )
    f1_upper_ridge_height: FloatProperty(
        name="F1 – Felső gerinc mag. (mm)", default=0.0, min=0.0, precision=2)
    f1_n_pairs: IntProperty(name="F1 – Mért pontpárok száma", default=0, min=0)
    # Felső gerincél 3D ívhossza (mm) – az F2 alámenősség méretfüggetlen
    # standardizálásához (V/L³). DB: F1_ivhossz_mm.
    f1_arc_length_mm: FloatProperty(
        name="F1 – Felső gerincél ívhossz (mm)", default=0.0, min=0.0, precision=2)
    # Raw profile points (JSON string) sent to the DB column F1_gerincelvonal
    f1_profile_json: StringProperty(default="", options={'HIDDEN'})

    f2_undercut_volume: FloatProperty(
        name="F2 – Alámenősség (mm³)", default=0.0, min=0.0, precision=2)
    f2_original_volume: FloatProperty(
        name="  Eredeti minta (mm³)", default=0.0, min=0.0, precision=2)
    f2_passive_volume: FloatProperty(
        name="  Kiblokkolt passzív minta (mm³)", default=0.0, min=0.0, precision=2)

    f3_palatal_vault: FloatProperty(
        name="F3 – Szájpad boltozata (mm)", default=0.0, min=0.0, precision=2)

    f4_angle_left: FloatProperty(
        name="F4 – Bal szög (°)", default=0.0, min=0.0, max=180.0, precision=1)
    f4_angle_right: FloatProperty(
        name="F4 – Jobb szög (°)", default=0.0, min=0.0, max=180.0, precision=1)

    f6_angle_left: FloatProperty(
        name="F6 – Bal szög (°)", default=0.0, min=0.0, max=90.0, precision=1)
    f6_angle_right: FloatProperty(
        name="F6 – Jobb szög (°)", default=0.0, min=0.0, max=90.0, precision=1)

    a2_method: EnumProperty(
        name="A2 módszer",
        items=[
            ('B', "b) Ortogonális (bukkális+linguális)", ""),
            ('C', "c) Ribbon (gerinc-ribbon + áthajlás-ribbon)", ""),
        ],
        default='C'
    )
    a2_lower_ridge_height: FloatProperty(
        # Nincs min korlát: az előjeles átlag lehet negatív is, ha a gerinc
        # a felület/bázis alá kerül (anatómiai jellegzetesség, nem hiba).
        name="A2 – Alsó gerinc mag. (mm)", default=0.0, precision=2)
    # Per-method results, kept so both can be stored side by side
    # (DB: A2_methodB / A2_methodC). 0 = not measured.
    a2_method_b: FloatProperty(name="A2 b) ortogonális (mm)", default=0.0, precision=2)
    a2_method_c: FloatProperty(name="A2 c) ribbon (mm)",      default=0.0, precision=2)
    # Raw profile points (JSON string) sent to the DB column A2_gerincelvonal
    a2_profile_json: StringProperty(default="", options={'HIDDEN'})

    a10_angle: FloatProperty(
        name="A10 – Állcsontreláció szög (°)",
        default=90.0, min=0.0, max=180.0, precision=1)

    csv_path: StringProperty(
        name="CSV fájl", default="", subtype='FILE_PATH')

    show_f2_detail: BoolProperty(name="Részletek", default=False)

class TAF_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ID

    server_url: StringProperty(
        name="Szerver URL",
        description="TAF adatszerver alap URL-je (pl. http://szerver:5002)",
        default="https://taf-hcax.onrender.com"
    )
    api_key: StringProperty(
        name="API kulcs",
        description="A szerver API kulcsa (BLENDER_API_KEY a .env-ben)",
        default="",
        subtype='PASSWORD'
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "server_url")
        layout.prop(self, "api_key")
        layout.label(text="A kulcs a Blender beállításaiban marad, nem kerül a .blend fájlba.", icon='LOCKED')


# ── Morphometria operators ────────────────────────────────────────────────────

class TAF_OT_FillPatientId(Operator):
    bl_idname = "taf.fill_patient_id"
    bl_label  = "ID a fájlnévből"

    def execute(self, context):
        filepath = bpy.data.filepath
        if filepath:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            context.scene.taf_props.patient_id = basename
            self.report({'INFO'}, f"TAJ: {basename}")
        else:
            self.report({'WARNING'}, "Mentsd el a .blend fájlt először!")
        return {'FINISHED'}


class TAF_OT_CalcF1(Operator):
    """Pontosan 2 mesh obj. kijelölve: gerincélvonal + bukkális áthajlás."""
    bl_idname = "taf.calc_f1"
    bl_label  = "Automatikus számítás (gerincél + bukkális áthajlás)"

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if len(selected) != 2:
            self.report({'ERROR'}, "Pontosan 2 mesh objektumot jelölj ki!")
            return {'CANCELLED'}
        avg_mm, n, raw_mm = avg_vertical_distance(selected[0], selected[1])
        if avg_mm is None:
            self.report({'ERROR'}, "Az objektumoknak nincs vertexük!")
            return {'CANCELLED'}
        props = context.scene.taf_props
        props.f1_upper_ridge_height = avg_mm
        props.f1_n_pairs = n
        props.f1_profile_json = _profile_json(raw_mm)
        # Felső gerincél ívhossza (selected[0] = gerinc az avg_vertical_distance-ban)
        props.f1_arc_length_mm = ridge_arclength_mm(selected[0])
        raw_file = _write_raw_distances_csv(
            bpy.path.abspath(props.csv_path), props.patient_id, "F1", raw_mm)
        suffix = f"  | ívhossz {props.f1_arc_length_mm:.1f} mm"
        if raw_file:
            self.report({'INFO'},
                f"F1 = {avg_mm:.2f} mm  ({n} pont){suffix} → nyers: {os.path.basename(raw_file)}")
        else:
            self.report({'INFO'}, f"F1 = {avg_mm:.2f} mm  ({n} pontpár){suffix}")
        return {'FINISHED'}


class TAF_OT_CalcA2(Operator):
    """A2 kétféleképp (a kijelölt objektumok száma dönt):
       b) 3 obj (gerinc + bukkális + linguális görbe) → ortogonális,
       c) 2 obj (gerinc-ribbon + egyesített áthajlás-ribbon) → ribbon."""
    bl_idname = "taf.calc_a2"
    bl_label  = "Automatikus számítás (gerincél + áthajlás(ok))"

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if len(selected) not in (2, 3):
            self.report({'ERROR'}, "Jelölj ki 2 vagy 3 objektumot!")
            return {'CANCELLED'}
        props    = context.scene.taf_props
        if len(selected) == 2:
            # c) Ribbon: gerinc-ribbon + egyesített áthajlás-ribbon felület.
            avg_mm, n, profile = a2_ribbon_profile(selected[0], selected[1])
            if avg_mm is None:
                self.report({'ERROR'},
                    "Nem sikerült a ribbon számítás (kevés vertex / nincs felület?).")
                return {'CANCELLED'}
            props.a2_lower_ridge_height = avg_mm
            props.a2_method = 'C'
            props.a2_method_c = avg_mm
            props.a2_profile_json = json.dumps(profile, ensure_ascii=False) if profile else ""
            self.report({'INFO'},
                f"A2 (c, ribbon) = {avg_mm:.2f} mm  ({n} keresztmetszeti pont)")
        else:
            # 3 objektum (gerincél + bukkális + linguális áthajlás):
            # keresztmetszeti, ortogonális szabványosított magasság (Method 2).
            avg_mm, n, profile = a2_orthogonal_profile(
                selected[0], selected[1], selected[2])
            if avg_mm is None:
                self.report({'ERROR'},
                    "Nem sikerült a Method 2 számítás (kevés vertex a görbéken?).")
                return {'CANCELLED'}
            props.a2_lower_ridge_height = avg_mm
            props.a2_method = 'B'
            props.a2_method_b = avg_mm
            props.a2_profile_json = json.dumps(profile, ensure_ascii=False) if profile else ""
            self.report({'INFO'},
                f"A2 (b) = {avg_mm:.2f} mm  ({n} keresztmetszeti pont, ortogonális)")
        return {'FINISHED'}


class TAF_OT_CalcF2(Operator):
    """Pontosan 2 mesh obj. kijelölve: eredeti + passzív blokkolt minta."""
    bl_idname = "taf.calc_f2"
    bl_label  = "Automatikus számítás (2 obj. kijel.)"

    def execute(self, context):
        selected = [o for o in context.selected_objects if o.type == 'MESH']
        if len(selected) != 2:
            self.report({'ERROR'}, "Pontosan 2 mesh objektumot jelölj ki!")
            return {'CANCELLED'}
        depsgraph = context.evaluated_depsgraph_get()
        vols = sorted([mesh_volume_mm3(o, depsgraph) for o in selected])
        props = context.scene.taf_props
        props.f2_original_volume  = vols[0]
        props.f2_passive_volume   = vols[1]
        props.f2_undercut_volume  = vols[1] - vols[0]
        self.report({'INFO'}, f"Alámenősség: {props.f2_undercut_volume:.2f} mm³")
        return {'FINISHED'}


class TAF_OT_CalcF3(Operator):
    """Aktív Single Vert objektum Z koordinátájából olvassa a boltozat magasságát."""
    bl_idname = "taf.calc_f3"
    bl_label  = "Z leolvasása (aktív Single Vert obj.)"

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy Single Vert objektumot!")
            return {'CANCELLED'}
        verts = obj.data.vertices
        if len(verts) == 0:
            self.report({'ERROR'}, "Az objektumnak nincsenek vertexei!")
            return {'CANCELLED'}
        world_z = max((obj.matrix_world @ v.co).z for v in verts)
        scale   = context.scene.unit_settings.scale_length or 1.0
        z_mm    = internal_to_mm(abs(world_z), scale)
        context.scene.taf_props.f3_palatal_vault = z_mm
        self.report({'INFO'}, f"Szájpad boltozat: {z_mm:.2f} mm")
        return {'FINISHED'}


class TAF_OT_ExportCSV(Operator):
    """Egy sort fűz a CSV fájlhoz az aktuális páciens adataival."""
    bl_idname = "taf.export_csv"
    bl_label  = "Sor hozzáadása a CSV-hez"

    def execute(self, context):
        props = context.scene.taf_props
        if not props.patient_id.strip():
            self.report({'ERROR'}, "Töltsd ki a TAJ számot!")
            return {'CANCELLED'}
        csv_path = bpy.path.abspath(props.csv_path)
        if not csv_path:
            self.report({'ERROR'}, "Add meg a CSV fájl útvonalát!")
            return {'CANCELLED'}
        f4_avg = (props.f4_angle_left  + props.f4_angle_right) / 2.0
        f6_avg = (props.f6_angle_left  + props.f6_angle_right) / 2.0
        angle  = props.a10_angle
        a10_class = (
            "Angle I"   if abs(angle - 90) < 5
            else ("Angle II" if angle > 90 else "Angle III")
        )
        row = {
            "taj_szam":                props.patient_id.strip(),
            "F1_felso_gerinc_mag_mm":  round(props.f1_upper_ridge_height, 2),
            "F2_alamenosseg_mm3":      round(props.f2_undercut_volume, 2),
            "F2_eredeti_mm3":          round(props.f2_original_volume, 2),
            "F2_passiv_mm3":           round(props.f2_passive_volume, 2),
            "F3_szajpad_boltozat_mm":  round(props.f3_palatal_vault, 2),
            "F4_szog_bal_fok":         round(props.f4_angle_left, 1),
            "F4_szog_jobb_fok":        round(props.f4_angle_right, 1),
            "F4_szog_atlag_fok":       round(f4_avg, 1),
            "F6_szog_bal_fok":         round(props.f6_angle_left, 1),
            "F6_szog_jobb_fok":        round(props.f6_angle_right, 1),
            "F6_szog_atlag_fok":       round(f6_avg, 1),
            "A2_modszer":              props.a2_method,
            "A2_also_gerinc_mag_mm":   round(props.a2_lower_ridge_height, 2),
            "A10_szog_fok":            round(props.a10_angle, 1),
            "A10_Angle_osztaly":       a10_class,
        }
        fieldnames  = list(row.keys())
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        self.report({'INFO'},
            f"Sor hozzáadva: {props.patient_id} → {os.path.basename(csv_path)}")
        return {'FINISHED'}


class TAF_OT_ClearFields(Operator):
    """Visszaállítja az összes mérési mezőt nullára (TAJ és CSV útvonal megmarad)."""
    bl_idname = "taf.clear_fields"
    bl_label  = "Mérések törlése"

    def execute(self, context):
        p = context.scene.taf_props
        for attr in (
            'f1_upper_ridge_height',
            'f2_undercut_volume', 'f2_original_volume', 'f2_passive_volume',
            'f3_palatal_vault',
            'f4_angle_left', 'f4_angle_right',
            'f6_angle_left', 'f6_angle_right',
            'a2_lower_ridge_height',
        ):
            setattr(p, attr, 0.0)
        p.a10_angle = 90.0
        self.report({'INFO'}, "Mezők törölve.")
        return {'FINISHED'}


class TAF_OT_AutoF3(Operator):
    """A fogászati modell mesh-ből automatikusan meghatározza a palatális boltozat
    magasságát: a középső régió legmagasabb Z-értéke = F3 (Z=0 az okklúziós sík)."""
    bl_idname  = "taf.auto_f3"
    bl_label   = "F3 auto (mesh)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy fogászati modell mesh-t!")
            return {'CANCELLED'}

        scale = context.scene.unit_settings.scale_length or 1.0

        # Edit módban: csak a kijelölt vertexek; Object módban: az összes
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            mat   = obj.matrix_world
            verts = [mat @ v.co.copy() for v in bm.verts if v.select]
            if not verts:
                self.report({'ERROR'},
                    "Nincs kijelölt vertex. Edit módban jelöld ki a palatális területet!")
                return {'CANCELLED'}
            source = "kijelölt vertexek"
        else:
            verts  = _world_vertices(obj)
            source = "összes vertex"

        if not verts:
            self.report({'ERROR'}, "Az objektumnak nincsenek vertexei!")
            return {'CANCELLED'}

        best = max(verts, key=lambda v: v.z)
        z_mm = internal_to_mm(best.z, scale)

        context.scene.taf_props.f3_palatal_vault = z_mm

        # Vizuális jelölő: gömb Empty a detektált ponton
        marker_name = "TAF_F3_Pont"
        if marker_name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[marker_name], do_unlink=True)
        marker = bpy.data.objects.new(marker_name, None)
        marker.empty_display_type = 'SPHERE'
        marker.empty_display_size = mm_to_internal(3.0, scale)
        marker.location           = best
        context.collection.objects.link(marker)

        self.report({'INFO'}, f"F3 auto: {z_mm:.2f} mm  ({source})  →  TAF_F3_Pont elhelyezve")
        return {'FINISHED'}


class TAF_OT_PlaceAngleMarker(Operator):
    """Kijelölt vertexek centroidjára lerak egy gömb-Empty jelölőt (szögméréshez)."""
    bl_idname  = "taf.place_angle_marker"
    bl_label   = "Szög jelölő lerak"
    bl_options = {'REGISTER', 'UNDO'}

    marker_name: StringProperty(default="TAF_Marker")

    def execute(self, context):
        pos, err = _place_marker_at_selection(context, self.marker_name)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        self.report({'INFO'}, f"{self.marker_name} → ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        return {'FINISHED'}


def _angle_to_horizontal(A, B):
    """Angle (deg) between segment A→B and the horizontal occlusal plane (Z=0)."""
    AB = B - A
    length = AB.length
    if length < 1e-6:
        return None
    return math.degrees(math.asin(abs(AB.z) / length))


class TAF_OT_CalcF6(Operator):
    """F6 – Interalveoláris szög: oldalanként a felső-alsó modell posteriori gerincélpontjait
    összekötő szakasz rágósíkkal (Z=0) bezárt szöge; a két oldal átlaga."""
    bl_idname  = "taf.calc_f6"
    bl_label   = "F6 kiszámítása"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.taf_props
        sides = (
            ("Bal",  _F6_BAL_FELSO_NAME,  _F6_BAL_ALSO_NAME,  "f6_angle_left"),
            ("Jobb", _F6_JOBB_FELSO_NAME, _F6_JOBB_ALSO_NAME, "f6_angle_right"),
        )
        results = []
        for label, felso, also, attr in sides:
            missing = [n for n in (felso, also) if n not in bpy.data.objects]
            if missing:
                self.report({'ERROR'}, f"{label} oldal – hiányzó jelölő: {', '.join(missing)}")
                return {'CANCELLED'}
            angle = _angle_to_horizontal(
                bpy.data.objects[felso].location,
                bpy.data.objects[also].location,
            )
            if angle is None:
                self.report({'ERROR'}, f"{label} oldal – a két pont azonos helyen van!")
                return {'CANCELLED'}
            setattr(props, attr, round(angle, 1))
            results.append(angle)
        avg = sum(results) / len(results)
        self.report({'INFO'},
            f"F6 interalveoláris: bal {results[0]:.1f}°, jobb {results[1]:.1f}° → átlag {avg:.1f}°")
        return {'FINISHED'}


class TAF_OT_CalcA10(Operator):
    """A10 – Állcsontreláció: felső és alsó frontális gerincélpont összekötője és a függőleges tengely szöge."""
    bl_idname  = "taf.calc_a10"
    bl_label   = "A10 kiszámítása"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        missing = [n for n in (_A10_FELSO_NAME, _A10_ALSO_NAME) if n not in bpy.data.objects]
        if missing:
            self.report({'ERROR'}, f"Hiányzó jelölő: {', '.join(missing)}")
            return {'CANCELLED'}
        A = bpy.data.objects[_A10_FELSO_NAME].location
        B = bpy.data.objects[_A10_ALSO_NAME].location
        AB = B - A
        length = AB.length
        if length < 1e-6:
            self.report({'ERROR'}, "A két pont azonos helyen van!")
            return {'CANCELLED'}
        # angle between line AB and the vertical axis (Z-axis)
        angle_deg = math.degrees(math.acos(min(1.0, abs(AB.z) / length)))
        context.scene.taf_props.a10_angle = round(angle_deg, 1)
        self.report({'INFO'}, f"A10 állcsontreláció szög: {angle_deg:.1f}°")
        return {'FINISHED'}


class TAF_OT_UploadToServer(Operator):
    """A mért értékeket TAJ szerint feltölti a remote adatbázisba."""
    bl_idname = "taf.upload_to_server"
    bl_label  = "Feltöltés a szerverre"

    def execute(self, context):
        import urllib.request
        import urllib.error
        import json as _json
        import time
        import datetime

        props = context.scene.taf_props
        preferences = _addon_preferences(context)

        if not props.patient_id.strip():
            self.report({'ERROR'}, "Töltsd ki a TAJ számot!")
            return {'CANCELLED'}
        if not preferences:
            self.report({'ERROR'}, "Az addon beállításai nem érhetők el. Telepítsd és engedélyezd az addont!")
            return {'CANCELLED'}
        if not preferences.server_url.strip():
            self.report({'ERROR'}, "Add meg a Szerver URL-t!")
            return {'CANCELLED'}
        if not preferences.api_key.strip():
            self.report({'ERROR'}, "Add meg az API kulcsot!")
            return {'CANCELLED'}

        f4_avg = (props.f4_angle_left  + props.f4_angle_right) / 2.0
        f6_avg = (props.f6_angle_left  + props.f6_angle_right) / 2.0

        def _parse_profile(s):
            if not s:
                return None
            try:
                return _json.loads(s)
            except Exception:
                return None

        # Warn (non-blocking) if a scalar value exists but its profile is missing —
        # usually means F1/A2 was measured before this addon version was loaded.
        missing_profiles = []
        if props.f1_upper_ridge_height > 0 and not props.f1_profile_json:
            missing_profiles.append("F1")
        if props.a2_lower_ridge_height > 0 and not props.a2_profile_json:
            missing_profiles.append("A2")
        if missing_profiles:
            self.report({'WARNING'},
                f"Nincs profil ehhez: {', '.join(missing_profiles)} — "
                f"futtasd újra a számítást a profil rögzítéséhez!")

        payload_dict = {
            "TAJ":        props.patient_id.strip(),
            "F1":         round(props.f1_upper_ridge_height, 2),
            "F1_ivhossz_mm": round(props.f1_arc_length_mm, 2) if props.f1_arc_length_mm else None,
            "F2":         round(props.f2_undercut_volume,    2),
            "F3":         round(props.f3_palatal_vault,      2),
            "F4":         round(f4_avg,                      1),
            "F6":         round(f6_avg,                      1),
            "A10":        round(props.a10_angle,             1),
            "A2_mag_mm":  round(props.a2_lower_ridge_height, 2),
            "A2_modszer": props.a2_method,
            # Per-method values (None → szerver megőrzi a meglévőt COALESCE-szal)
            "A2_methodB": round(props.a2_method_b, 2) if props.a2_method_b else None,
            "A2_methodC": round(props.a2_method_c, 2) if props.a2_method_c else None,
            # Raw profile point arrays → DB columns F1_profil / A2_profil
            "F1_profil": _parse_profile(props.f1_profile_json),
            "A2_profil": _parse_profile(props.a2_profile_json),
        }
        payload = _json.dumps(payload_dict).encode('utf-8')

        # Local backup before uploading
        backup_path = None
        csv_dir = (os.path.dirname(bpy.path.abspath(props.csv_path))
                   if props.csv_path else "")
        if csv_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                csv_dir,
                f"{props.patient_id.strip()}_{ts}_upload.json"
            )
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    _json.dump(payload_dict, f, ensure_ascii=False, indent=2)
            except Exception:
                backup_path = None  # non-fatal; proceed without backup

        url = preferences.server_url.rstrip('/') + '/api/morphometria'
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'X-API-Key':    preferences.api_key.strip(),
            },
            method='POST'
        )

        last_error = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = _json.loads(resp.read().decode('utf-8'))
                if result.get('error'):
                    self.report({'ERROR'}, f"Szerver elutasítás: {result['error']}")
                    return {'CANCELLED'}
                # Sikeres feltöltés — töröljük a biztonsági mentést
                if backup_path and os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                    except Exception:
                        pass
                suffix = f" ({attempt}. kísérlet)" if attempt > 1 else ""
                self.report({'INFO'}, f"Feltöltve: {props.patient_id}{suffix}")
                return {'FINISHED'}
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8')
                try:
                    msg = _json.loads(body).get('error', body)
                except Exception:
                    msg = body
                self.report({'ERROR'}, f"Szerver hiba ({e.code}): {msg}")
                return {'CANCELLED'}
            except (urllib.error.URLError, OSError) as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
            if attempt < 3:
                time.sleep(2 * attempt)  # 2s, majd 4s

        backup_hint = (f" (backup: {os.path.basename(backup_path)})"
                       if backup_path else "")
        self.report({'ERROR'},
            f"Hálózati hiba 3 kísérlet után: {last_error}{backup_hint}")
        return {'CANCELLED'}


class TAF_OT_SaveAndUploadBlend(Operator):
    """Save the project, upload measurements, then archive the .blend file."""
    bl_idname = "taf.save_and_upload_blend"
    bl_label = "Mentés + adatok és .blend feltöltése"
    bl_options = {'REGISTER'}

    def execute(self, context):
        import http.client
        import json as _json
        import time
        import urllib.parse
        import uuid

        props = context.scene.taf_props
        preferences = _addon_preferences(context)
        patient_id = props.patient_id.strip()

        if not patient_id:
            self.report({'ERROR'}, "Töltsd ki a TAJ számot!")
            return {'CANCELLED'}
        if not bpy.data.filepath:
            self.report({'ERROR'}, "Először mentsd el a .blend fájlt a gépen!")
            return {'CANCELLED'}
        if not preferences or not preferences.server_url.strip() or not preferences.api_key.strip():
            self.report({'ERROR'}, "Add meg a szerver URL-jét és az API kulcsot az addon beállításaiban!")
            return {'CANCELLED'}

        try:
            bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
        except Exception as exc:
            self.report({'ERROR'}, f"A .blend mentése nem sikerült: {exc}")
            return {'CANCELLED'}

        measurement_result = bpy.ops.taf.upload_to_server()
        if 'FINISHED' not in measurement_result:
            self.report({'ERROR'}, "A mérési adatok feltöltése sikertelen; a .blend nem lett feltöltve.")
            return {'CANCELLED'}

        filepath = bpy.data.filepath
        file_size = os.path.getsize(filepath)
        endpoint = preferences.server_url.rstrip('/') + '/api/morphometria/blend'
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            self.report({'ERROR'}, "Érvénytelen szerver URL.")
            return {'CANCELLED'}

        boundary = f"----TAFBlend{uuid.uuid4().hex}"
        patient_part = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="TAJ"\r\n\r\n'
            f"{patient_id}\r\n"
        ).encode('utf-8')
        file_part = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="blend_file"; filename="model.blend"\r\n'
            'Content-Type: application/x-blender\r\n\r\n'
        ).encode('ascii')
        closing = f"\r\n--{boundary}--\r\n".encode('ascii')
        content_length = len(patient_part) + len(file_part) + file_size + len(closing)

        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == 'https'
            else http.client.HTTPConnection
        )

        last_error = None
        for attempt in range(1, 4):
            connection = connection_class(parsed.hostname, parsed.port, timeout=900)
            try:
                connection.putrequest('POST', path)
                connection.putheader('Content-Type', f'multipart/form-data; boundary={boundary}')
                connection.putheader('Content-Length', str(content_length))
                connection.putheader('X-API-Key', preferences.api_key.strip())
                connection.endheaders()
                connection.send(patient_part)
                connection.send(file_part)
                with open(filepath, 'rb') as blend_file:
                    while True:
                        chunk = blend_file.read(1024 * 1024)
                        if not chunk:
                            break
                        connection.send(chunk)
                connection.send(closing)
                response = connection.getresponse()
                body = response.read().decode('utf-8', errors='replace')
                try:
                    result = _json.loads(body)
                except Exception:
                    result = {}
                if response.status < 200 or response.status >= 300:
                    message = result.get('error') or body or response.reason
                    self.report({'ERROR'}, f".blend feltöltési hiba ({response.status}): {message}")
                    return {'CANCELLED'}
                remote_name = result.get('filename', 'a páciens mappájába')
                suffix = f" ({attempt}. kísérlet)" if attempt > 1 else ""
                self.report({'INFO'}, f"Adatok és .blend feltöltve: {remote_name}{suffix}")
                return {'FINISHED'}
            except (OSError, http.client.HTTPException) as exc:
                last_error = str(exc)
            finally:
                connection.close()
            if attempt < 3:
                time.sleep(2 * attempt)

        self.report({'ERROR'}, f"A mérési adatok megvannak, de a .blend feltöltése sikertelen: {last_error}")
        return {'CANCELLED'}


class TAF_OT_AddCursorVertexProbe(Operator):
    """Egyetlen vertex a 3D kurzornál, modifier-stackkel:
       Subdivision (3) → Shrinkwrap (a gomb megnyomásakor kijelölt mesh-re,
       a felület felett, 0.2 mm offset) → Smooth (edit módban és cage-en, factor 4).
    A cél mesh-t a gomb megnyomása ELŐTT jelöld ki (aktív objektum)."""
    bl_idname  = "taf.add_cursor_vertex_probe"
    bl_label   = "Vertex-szonda a kurzornál"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # A gomb megnyomásakor kijelölt cél mesh (aktív, vagy az első kijelölt).
        target = context.active_object
        if target is None or target.type != 'MESH':
            target = next((o for o in context.selected_objects if o.type == 'MESH'), None)
        if target is None:
            self.report({'ERROR'}, "Jelölj ki egy cél mesh-t a gomb megnyomása előtt!")
            return {'CANCELLED'}

        scale = context.scene.unit_settings.scale_length or 1.0
        cursor_loc = context.scene.cursor.location.copy()

        # Egyetlen vertexes mesh a 3D kurzornál.
        mesh = bpy.data.meshes.new("TAF_VertexProbe")
        mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
        mesh.update()
        obj = bpy.data.objects.new("TAF_VertexProbe", mesh)
        obj.location = cursor_loc
        context.collection.objects.link(obj)

        # 1) Subdivision Surface, level 3
        sub = obj.modifiers.new("Subdivision", 'SUBSURF')
        sub.levels = 3
        sub.render_levels = 3

        # 2) Shrinkwrap a cél mesh-re, a felület felett, 0.2 mm offset
        sw = obj.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
        sw.target = target
        sw.wrap_method = 'NEAREST_SURFACEPOINT'
        sw.wrap_mode = 'ABOVE_SURFACE'
        sw.offset = mm_to_internal(0.2, scale)

        # 3) Smooth: edit módban és cage-en látható, factor 4
        sm = obj.modifiers.new("Smooth", 'SMOOTH')
        sm.factor = 4.0
        sm.show_in_editmode = True
        sm.show_on_cage = True

        # Az új objektum legyen az egyetlen kijelölt + aktív.
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({'INFO'}, f"Vertex-szonda létrehozva (cél: {target.name})")
        return {'FINISHED'}


# ── Morphometria panel ────────────────────────────────────────────────────────

class TAF_PT_Main(Panel):
    bl_label       = "TAF Morphometria"
    bl_idname      = "TAF_PT_main"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'TAF'

    def draw(self, context):
        layout = self.layout
        props  = context.scene.taf_props

        box = layout.box()
        box.label(text="Páciens", icon='USER')
        row = box.row(align=True)
        row.prop(props, "patient_id", text="TAJ")
        row.operator("taf.fill_patient_id", text="", icon='FILE_BLEND')

        box = layout.box()
        box.label(text="F1  Felső gerinc magassága", icon='SORTSIZE')
        box.label(text="Jelöld ki: gerincélvonal + bukkális áthajlás", icon='INFO')
        box.operator("taf.calc_f1", icon='DRIVER_DISTANCE')
        row = box.row()
        row.prop(props, "f1_upper_ridge_height", text="Átlag (mm)")
        if props.f1_n_pairs > 0:
            row.label(text=f"n={props.f1_n_pairs}")
        box.prop(props, "f1_arc_length_mm", text="Ívhossz L (mm)")

        box = layout.box()
        box.label(text="F2  Alámenős területek", icon='MOD_BOOLEAN')
        box.operator("taf.calc_f2", icon='SNAP_VOLUME')
        row = box.row()
        row.prop(props, "show_f2_detail",
                 icon='TRIA_DOWN' if props.show_f2_detail else 'TRIA_RIGHT',
                 emboss=False, text="Részletek")
        if props.show_f2_detail:
            box.prop(props, "f2_original_volume", text="Eredeti (mm³)")
            box.prop(props, "f2_passive_volume",  text="Passzív (mm³)")
        box.prop(props, "f2_undercut_volume", text="Alámenősség (mm³)")

        box = layout.box()
        box.label(text="F3  Szájpad boltozata", icon='OBJECT_DATA')
        row = box.row(align=True)
        row.operator("taf.auto_f3",  icon='ZOOM_SELECTED', text="Auto (mesh)")
        row.operator("taf.calc_f3",  icon='VERTEXSEL',     text="Single Vert-ből")
        box.prop(props, "f3_palatal_vault", text="Magasság (mm)")

        box = layout.box()
        box.label(text="F4  Felső gerinc alakja (szög)",
                  icon='DRIVER_ROTATIONAL_DIFFERENCE')
        col = box.column(align=True)
        col.prop(props, "f4_angle_left",  text="Bal (°)")
        col.prop(props, "f4_angle_right", text="Jobb (°)")
        box.label(text=f"Átlag: {(props.f4_angle_left + props.f4_angle_right) / 2:.1f}°")

        box = layout.box()
        box.label(text="F6  Interalveoláris / rágósík szög",
                  icon='DRIVER_ROTATIONAL_DIFFERENCE')
        box.label(text="Edit módban jelöld a posteriori gerincélpontot, majd a gombot",
                  icon='INFO')
        col = box.column(align=True)
        col.label(text="Bal oldal:")
        row = col.row(align=True)
        op = row.operator("taf.place_angle_marker", text="Felső", icon='CURSOR')
        op.marker_name = _F6_BAL_FELSO_NAME
        op = row.operator("taf.place_angle_marker", text="Alsó", icon='CURSOR')
        op.marker_name = _F6_BAL_ALSO_NAME
        col.label(text="Jobb oldal:")
        row = col.row(align=True)
        op = row.operator("taf.place_angle_marker", text="Felső", icon='CURSOR')
        op.marker_name = _F6_JOBB_FELSO_NAME
        op = row.operator("taf.place_angle_marker", text="Alsó", icon='CURSOR')
        op.marker_name = _F6_JOBB_ALSO_NAME
        box.operator("taf.calc_f6", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        f6_avg = (props.f6_angle_left + props.f6_angle_right) / 2.0
        box.label(text=f"Bal: {props.f6_angle_left:.1f}°  Jobb: {props.f6_angle_right:.1f}°  →  átlag {f6_avg:.1f}°")

        box = layout.box()
        box.label(text="A2  Alsó gerinc magassága", icon='SORTSIZE')
        box.label(text="2 obj → c) ribbon | 3 obj → b) ortogonális", icon='INFO')
        box.operator("taf.calc_a2", icon='DRIVER_DISTANCE')
        box.prop(props, "a2_method", text="Módszer (auto)")
        box.prop(props, "a2_lower_ridge_height", text="Aktív (mm)")
        col = box.column(align=True)
        col.label(text="Tárolt módszerenként (0 = nincs mérve):")
        col.prop(props, "a2_method_b", text="b) ortogonális")
        col.prop(props, "a2_method_c", text="c) ribbon")

        box = layout.box()
        box.label(text="A10  Állcsontreláció", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        box.label(text="Edit módban jelöld ki a felső, majd az alsó frontális gerincélpontot",
                  icon='INFO')
        row = box.row(align=True)
        op = row.operator("taf.place_angle_marker", text="Felső pont", icon='CURSOR')
        op.marker_name = _A10_FELSO_NAME
        op = row.operator("taf.place_angle_marker", text="Alsó pont", icon='CURSOR')
        op.marker_name = _A10_ALSO_NAME
        box.operator("taf.calc_a10", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        angle = props.a10_angle
        cls   = ("Angle I"   if abs(angle - 90) < 5
                 else ("Angle II" if angle > 90 else "Angle III"))
        box.label(text=f"Szög: {angle:.1f}°  →  {cls}", icon='FUND')

        layout.separator()
        box = layout.box()
        box.label(text="Eszközök", icon='TOOL_SETTINGS')
        box.label(text="Jelöld ki a cél mesh-t, állítsd a 3D kurzort, majd:", icon='INFO')
        box.operator("taf.add_cursor_vertex_probe", icon='OUTLINER_OB_POINTCLOUD')

        box = layout.box()
        box.label(text="Export", icon='EXPORT')
        box.prop(props, "csv_path", text="CSV")
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator("taf.export_csv",   icon='FILE_TICK')
        row.operator("taf.clear_fields", icon='TRASH', text="Töröl")

        layout.separator()
        box = layout.box()
        box.label(text="Remote adatbázis", icon='URL')
        preferences = _addon_preferences(context)
        if preferences:
            box.prop(preferences, "server_url", text="URL")
            box.prop(preferences, "api_key", text="API kulcs")
        else:
            box.label(text="Állítsd be az addont a Blender Preferences ablakban.", icon='ERROR')
        row = box.row(align=True)
        row.scale_y = 1.4
        row.operator("taf.upload_to_server", icon='EXPORT', text="Csak adatok")
        row = box.row()
        row.scale_y = 1.6
        row.operator("taf.save_and_upload_blend", icon='FILE_BLEND')
        box.label(text="Helyben ment, majd verziózva archiválja a NAS-on.", icon='INFO')


# ===========================================================================
# ── CURVE DETECTOR ────────────────────────────────────────────────────────────
# ===========================================================================

class TAF_CD_Props(PropertyGroup):

    jaw_type: EnumProperty(
        name="Állcsont",
        items=[
            ('UPPER', "Felső (maxilla)",  "Felső fogsor nélküli modell – okkluzális felfelé (+Z)"),
            ('LOWER', "Alsó (mandibula)", "Alsó fogsor nélküli modell – ortogonális magasságmérés"),
        ],
        default='UPPER'
    )
    n_slices: IntProperty(
        name="Szeletek száma", default=30, min=6, max=120)
    margin_pct: FloatProperty(
        name="Határ levágás (%)", default=5.0, min=0.0, max=30.0, subtype='PERCENTAGE')
    min_sulcus_drop: FloatProperty(
        name="Min. áthajlás mélység (mm)", default=0.5, min=0.0, max=15.0)
    arch_axis: EnumProperty(
        name="Ívirány",
        items=[
            ('ARCH',  "Ívközép (auto)", "Automatikus ívközépvonal (AJÁNLOTT)"),
            ('X',     "X tengely",      "Párhuzamos szeletek YZ síkban"),
            ('Y',     "Y tengely",      "Párhuzamos szeletek XZ síkban"),
            ('CURVE', "Arch_Axis",      "\"Arch_Axis\" nevű kézzel rajzolt görbe"),
        ],
        default='ARCH'
    )
    crest_window_mm: FloatProperty(
        name="Gerincél keresési ablak (mm)", default=4.0, min=0.5, max=20.0)
    fold_riseback_mm: FloatProperty(
        name="Áthajlás visszahajlás (mm)", default=0.4, min=0.0, max=5.0)
    use_ridge_region: BoolProperty(
        name="Ívillesztés a gerincrégióból", default=True)
    ridge_region_pct: FloatProperty(
        name="Gerincrégió (felső Z %)", default=45.0, min=10.0, max=100.0, subtype='PERCENTAGE')
    reject_border: BoolProperty(
        name="Mesh-szél kizárása", default=True)
    smooth_strength: IntProperty(
        name="Pontsor simítás", default=2, min=0, max=6)
    replace_existing: BoolProperty(
        name="Meglévők felülírása", default=True)
    smooth_curve: BoolProperty(
        name="Görbe simítása (NURBS rend)", default=True)
    curve_resolution: IntProperty(
        name="Görbe felbontás", default=12, min=1, max=64)
    show_advanced: BoolProperty(name="Haladó beállítások", default=False)


# ── Curve Detector operators ──────────────────────────────────────────────────

class TAF_OT_DetectCurves(Operator):
    """Automatikusan detektálja a gerincélt és az áthajlásokat
    a kijelölt fogászati modellen keresztmetszeti szeletelés alapján."""
    bl_idname  = "taf.detect_curves"
    bl_label   = "Gerincél / Áthajlás detektálása"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy fogászati modell mesh objektumot!")
            return {'CANCELLED'}

        props       = context.scene.taf_cd_props
        scene_scale = context.scene.unit_settings.scale_length
        is_lower    = (props.jaw_type == 'LOWER')
        reject_b    = props.reject_border

        frames, _ = get_slice_frames(obj, props)
        if not frames:
            self.report({'ERROR'}, "Nem sikerült szeletelési kereteket meghatározni.")
            return {'CANCELLED'}

        crest_pts   = []
        buccal_pts  = []
        lingual_pts = []
        ortho_dists = []
        empty_slices = 0
        no_landmark  = 0

        for fr in frames:
            raw = slice_mesh_edges(obj, fr.co, fr.normal, reject_border=reject_b)
            if len(raw) < (5 if is_lower else 4):
                empty_slices += 1
                continue

            if is_lower:
                crest, buc, lin, odist = detect_lower(raw, fr, props, scene_scale)
                if crest is None:
                    no_landmark += 1
                    continue
                crest_pts.append(crest)
                buccal_pts.append(buc)
                lingual_pts.append(lin)
                ortho_dists.append(odist)
            else:
                crest, sulcus = detect_upper(raw, fr, props, scene_scale)
                if crest is None:
                    empty_slices += 1
                    continue
                crest_pts.append(crest)
                if sulcus is not None:
                    buccal_pts.append(sulcus)
                else:
                    no_landmark += 1

        if len(crest_pts) < 4:
            self.report({'ERROR'},
                f"Csak {len(crest_pts)} gerincél pont detektálható "
                f"({empty_slices} üres szelet). Csökkentsd a Határ levágást, "
                "növeld a Gerincél keresési ablakot, vagy ellenőrizd az orientációt (+Z fel).")
            return {'CANCELLED'}

        s           = props.smooth_strength
        crest_pts   = _smooth_sequence(crest_pts,   s)
        buccal_pts  = _smooth_sequence(buccal_pts,  s)
        lingual_pts = _smooth_sequence(lingual_pts, s)

        res    = props.curve_resolution
        smooth = props.smooth_curve

        crest_obj = make_nurbs_curve(CREST_NAME, crest_pts, res, smooth)
        crest_obj.data.bevel_depth = 0.0002
        _assign_material(crest_obj, "TAF_Gerincelvonal_Mat", (0.0, 0.25, 1.0, 1.0))

        jaw_label = "Alsó" if is_lower else "Felső"
        extra_msg = ""

        if is_lower:
            if len(buccal_pts) >= 4:
                buc_obj = make_nurbs_curve(BUCCAL_NAME, buccal_pts, res, smooth)
                buc_obj.data.bevel_depth = 0.0002
                _assign_material(buc_obj, "TAF_Athajlas_Mat", (1.0, 0.4, 0.0, 1.0))
            else:
                self.report({'WARNING'},
                    f"Csak {len(buccal_pts)} bukkális áthajlás pont – rajzold kézzel.")
            if len(lingual_pts) >= 4:
                ling_obj = make_nurbs_curve(LINGUAL_NAME, lingual_pts, res, smooth)
                ling_obj.data.bevel_depth = 0.0002
                _assign_material(ling_obj, "TAF_Lingualis_Mat", (0.0, 0.8, 0.2, 1.0))
            else:
                self.report({'WARNING'},
                    f"Csak {len(lingual_pts)} linguális áthajlás pont – rajzold kézzel.")
            if ortho_dists:
                avg_mm = internal_to_mm(sum(ortho_dists) / len(ortho_dists), scene_scale)
                extra_msg = (f", ortogonális mag.: {len(ortho_dists)} szelet, "
                             f"átlag {avg_mm:.2f} mm")
        else:
            if len(buccal_pts) >= 4:
                sulcus_obj = make_nurbs_curve(BUCCAL_NAME, buccal_pts, res, smooth)
                sulcus_obj.data.bevel_depth = 0.0002
                _assign_material(sulcus_obj, "TAF_Athajlas_Mat", (1.0, 0.4, 0.0, 1.0))
                extra_msg = f", áthajlás: {len(buccal_pts)} pont"
            else:
                self.report({'WARNING'},
                    f"Csak {len(buccal_pts)} érvényes áthajlás pont "
                    f"({no_landmark} szelet kihagyva). Csökkentsd a Min. áthajlás mélységet "
                    "vagy az Áthajlás visszahajlást, vagy rajzold kézzel.")
                extra_msg = " | áthajlás: kézzel szükséges"

        self.report({'INFO'},
            f"Kész ({jaw_label}). Gerincél: {len(crest_pts)} pont{extra_msg}")
        return {'FINISHED'}


class TAF_OT_PreviewAxis(Operator):
    """'TAF_Arch_Preview' segédgörbét hoz létre az ívközépvonalból."""
    bl_idname  = "taf.preview_axis"
    bl_label   = "Ívközép előnézet"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy mesh objektumot!")
            return {'CANCELLED'}
        frames, _ = get_slice_frames(obj, context.scene.taf_cd_props)
        if not frames:
            self.report({'ERROR'}, "Nem sikerült ívközépvonalat illeszteni.")
            return {'CANCELLED'}
        prev = make_nurbs_curve("TAF_Arch_Preview", [fr.co for fr in frames], 12, True)
        prev.data.bevel_depth = 0.0003
        _assign_material(prev, "TAF_ArchPreview_Mat", (1.0, 1.0, 0.0, 1.0))
        self.report({'INFO'}, f"Ívközép előnézet: {len(frames)} szeletpont.")
        return {'FINISHED'}


class TAF_OT_ShowArchAxis(Operator):
    """Befoglaló doboz méreteit és az aktuális módot írja ki az Info sávba."""
    bl_idname  = "taf.show_arch_axis"
    bl_label   = "Tengely ellenőrzése"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy mesh objektumot!")
            return {'CANCELLED'}
        xs, ys, zs = _bbox_world(obj)
        self.report({'INFO'},
            f"BBox X:{min(xs):.1f}…{max(xs):.1f} ({max(xs)-min(xs):.1f})  "
            f"Y:{min(ys):.1f}…{max(ys):.1f} ({max(ys)-min(ys):.1f})  "
            f"Z:{min(zs):.1f}…{max(zs):.1f}  │  "
            f"Mód: {context.scene.taf_cd_props.arch_axis}  │  okkluzális irány = +Z legyen")
        return {'FINISHED'}


# ── Curve Detector panel ──────────────────────────────────────────────────────

class TAF_PT_CurveDetector(Panel):
    bl_label       = "Görbe Detektor"
    bl_idname      = "TAF_PT_curve_detector"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'TAF'
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props  = context.scene.taf_cd_props

        box0 = layout.box()
        box0.label(text="Állcsont típus", icon='BONE_DATA')
        box0.prop(props, "jaw_type", expand=True)
        if props.jaw_type == 'LOWER':
            info = box0.box()
            info.label(text="Mód: ortogonális magasságmérés", icon='DRIVER_DISTANCE')
            info.label(text="(Linguális + bukkális + gerincél)", icon='BLANK1')

        box = layout.box()
        box.label(text="Szeletelés", icon='SETTINGS')
        col = box.column(align=True)
        col.prop(props, "arch_axis",  text="Ívirány")
        col.prop(props, "n_slices",   text="Szeletek száma")
        col.prop(props, "margin_pct", text="Határ levágás")
        col.prop(props, "min_sulcus_drop",
                 text="Min. ortogonális magasság (mm)" if props.jaw_type == 'LOWER'
                 else "Min. áthajlás mélység (mm)")
        if props.arch_axis == 'ARCH':
            box.operator("taf.preview_axis", icon='CURVE_PATH', text="Ívközép előnézet")
        elif props.arch_axis == 'CURVE':
            box.box().label(text="\"Arch_Axis\" görbe szükséges", icon='CURVE_BEZCURVE')

        boxd = layout.box()
        boxd.label(text="Detektálás", icon='VIEWZOOM')
        boxd.column(align=True).prop(props, "crest_window_mm", text="Gerincél ablak (mm)")
        boxd.column(align=True).prop(props, "smooth_strength", text="Pontsor simítás")
        boxd.prop(props, "show_advanced",
                  text="Haladó beállítások",
                  icon='TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT', emboss=False)
        if props.show_advanced:
            adv = boxd.column(align=True)
            adv.prop(props, "fold_riseback_mm", text="Áthajlás visszahajlás (mm)")
            adv.prop(props, "reject_border",    text="Mesh-szél kizárása")
            adv.prop(props, "use_ridge_region", text="Ívillesztés gerincrégióból")
            if props.use_ridge_region:
                adv.prop(props, "ridge_region_pct", text="Gerincrégió (felső Z %)")

        sub = layout.box()
        sub.label(text="Görbe megjelenítés", icon='CURVE_DATA')
        row = sub.row(align=True)
        row.prop(props, "smooth_curve",     text="NURBS simítás")
        row.prop(props, "curve_resolution", text="Felbontás")
        sub.prop(props, "replace_existing", text="Meglévők felülírása")

        layout.separator()
        box2 = layout.box()
        box2.label(text="Kijelölés: fogászati modell mesh", icon='INFO')
        box2.label(text="Okkluzális irány = +Z (felfelé)", icon='ORIENTATION_NORMAL')
        col2 = layout.column(align=True)
        col2.scale_y = 1.5
        col2.operator("taf.detect_curves",  icon='MOD_MESHDEFORM')
        col2.operator("taf.show_arch_axis", icon='EMPTY_ARROWS',
                      text="Tengely / BBox ellenőrzés")

        layout.separator()
        box3 = layout.box()
        box3.label(text="Eredmény objektumok", icon='OUTLINER_OB_CURVE')
        has_c = CREST_NAME   in bpy.data.objects
        has_b = BUCCAL_NAME  in bpy.data.objects
        has_l = LINGUAL_NAME in bpy.data.objects
        is_low = (props.jaw_type == 'LOWER')
        for name, present in [(CREST_NAME, has_c), (BUCCAL_NAME, has_b)]:
            row = box3.row()
            row.label(text=("✓  " if present else "–  ") + name,
                      icon='CHECKMARK' if present else 'X')
        if is_low:
            row = box3.row()
            row.label(text=("✓  " if has_l else "–  ") + LINGUAL_NAME,
                      icon='CHECKMARK' if has_l else 'X')
            if has_c and has_b and has_l:
                box3.label(text="Kész az F1 méréshez (Morphometria › CalcF1)", icon='INFO')
        else:
            if has_c and has_b:
                box3.label(text="Kész az F1 méréshez (Morphometria › CalcF1)", icon='INFO')


# ===========================================================================
# ── CURVE LEARNER ─────────────────────────────────────────────────────────────
# ===========================================================================

class TAF_ML_Props(PropertyGroup):

    jaw_type: EnumProperty(
        name="Állcsont",
        items=[
            ('UPPER', "Felső (maxilla)",  "Felső modell – +Z felfelé"),
            ('LOWER', "Alsó (mandibula)", "Alsó modell – +Z felfelé"),
        ],
        default='UPPER'
    )
    n_slices: IntProperty(
        name="Szeletek száma", default=30, min=8, max=120)
    margin_pct: FloatProperty(
        name="Határ levágás (%)", default=5.0, min=0.0, max=30.0, subtype='PERCENTAGE')
    arch_axis: EnumProperty(
        name="Ívirány",
        items=[
            ('ARCH',  "Ívközép (auto)", "Automatikus ívközépvonal (AJÁNLOTT)"),
            ('X',     "X tengely",      "Párhuzamos szeletek YZ síkban"),
            ('Y',     "Y tengely",      "Párhuzamos szeletek XZ síkban"),
            ('CURVE', "Arch_Axis",      "\"Arch_Axis\" görbe tangense"),
        ],
        default='ARCH'
    )
    use_ridge_region: BoolProperty(name="Ívillesztés gerincrégióból", default=True)
    ridge_region_pct: FloatProperty(
        name="Gerincrégió (felső Z %)", default=45.0, min=10.0, max=100.0, subtype='PERCENTAGE')

    feat_samples: IntProperty(
        name="Profil mintavétel (K)", default=16, min=6, max=40)
    ridge_lambda: FloatProperty(
        name="Regularizáció (λ)", default=5.0, min=0.0, max=1000.0)
    snap_mm: FloatProperty(
        name="Felület-snap ablak (mm)", default=1.5, min=0.0, max=10.0)
    smooth_strength: IntProperty(
        name="Pontsor simítás", default=2, min=0, max=6)

    dataset_path: StringProperty(
        name="Tanító adat (JSON)", subtype='FILE_PATH', default="//taf_training_data.json")
    model_path: StringProperty(
        name="Modell (JSON)", subtype='FILE_PATH', default="//taf_curve_model.json")

    init_offset_mm: FloatProperty(
        name="Felület-offset (mm)", default=0.0, min=0.0, max=3.0)
    init_surface_draw: BoolProperty(name="Felületre rajzolás mód", default=True)

    smooth_curve: BoolProperty(name="NURBS simítás", default=True)
    curve_resolution: IntProperty(name="Felbontás", default=12, min=1, max=64)
    show_advanced: BoolProperty(name="Haladó", default=False)


# ── Curve Learner operators ───────────────────────────────────────────────────

class TAF_ML_OT_InitCurve(Operator):
    """Helyesen elnevezett, felülethez tapadó annotációs görbét hoz létre."""
    bl_idname  = "taf_ml.init_curve"
    bl_label   = "Görbe iniciátor"
    bl_options = {'REGISTER', 'UNDO'}

    landmark: EnumProperty(
        name="Landmark",
        items=[
            ('crest',   "Gerincél",           ""),
            ('buccal',  "Bukkális áthajlás",  ""),
            ('lingual', "Linguális áthajlás", ""),
        ],
        default='crest'
    )

    def execute(self, context):
        mesh = context.active_object
        if mesh is None or mesh.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki a rajzfelületnek egy mesh modellt!")
            return {'CANCELLED'}
        props  = context.scene.taf_ml_props
        scale  = context.scene.unit_settings.scale_length
        name   = LM_CURVE[self.landmark]
        bevel  = mm_to_internal(0.15, scale)
        offset = mm_to_internal(props.init_offset_mm, scale)

        cobj = bpy.data.objects.get(name)
        if cobj is not None and cobj.type != 'CURVE':
            self.report({'ERROR'},
                f"Létezik '{name}' nevű, de nem görbe objektum. Nevezd át / töröld.")
            return {'CANCELLED'}
        if cobj is None:
            cdata = bpy.data.curves.new(name, type='CURVE')
            cdata.dimensions = '3D'
            cobj = bpy.data.objects.new(name, cdata)
            context.collection.objects.link(cobj)
            cobj.matrix_world = mesh.matrix_world.copy()

        cobj.data.dimensions  = '3D'
        cobj.data.bevel_depth = bevel
        _assign_material(cobj, *LM_COLORS[self.landmark])

        mod = next((m for m in cobj.modifiers if m.type == 'SHRINKWRAP'), None)
        if mod is None:
            mod = cobj.modifiers.new("TAF_Shrinkwrap", 'SHRINKWRAP')
        mod.target       = mesh
        mod.wrap_method  = 'NEAREST_SURFACEPOINT'
        mod.offset       = offset

        for o in context.selected_objects:
            o.select_set(False)
        cobj.select_set(True)
        context.view_layer.objects.active = cobj

        hint = ""
        if props.init_surface_draw:
            ok   = self._enter_surface_draw(context, offset)
            hint = (" → Rajzolj a felületre (Draw eszköz aktív)."
                    if ok else
                    " → Lépj Edit módba, válaszd a Draw eszközt, Depth=Surface.")
        self.report({'INFO'}, f"'{name}' görbe kész, felülethez tapad.{hint}")
        return {'FINISHED'}

    def _enter_surface_draw(self, context, offset_internal):
        ok = True
        try:
            if context.object.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
        except Exception:
            ok = False
        ts = context.scene.tool_settings
        try:
            cps = ts.curve_paint_settings
            cps.depth_mode            = 'SURFACE'
            cps.use_offset_absolute   = True
            cps.surface_offset        = offset_internal
            cps.use_project_only_selected = False
        except Exception:
            pass
        try:
            ts.use_snap      = True
            ts.snap_elements = {'FACE'}
        except Exception:
            pass
        for attr in ("use_snap_project", "use_snap_self"):
            try: setattr(ts, attr, True)
            except Exception: pass
        try:
            area = next((a for a in context.screen.areas if a.type == 'VIEW_3D'), None)
            if area and hasattr(context, "temp_override"):
                with context.temp_override(area=area):
                    bpy.ops.wm.tool_set_by_id(name="builtin.draw")
            else:
                ok = False
        except Exception:
            ok = False
        return ok


class TAF_ML_OT_AddSample(Operator):
    """Szeletmintákat fűz a tanító készlethez az aktuális annotált modellből."""
    bl_idname  = "taf_ml.add_sample"
    bl_label   = "Hozzáadás a tanító készlethez"
    bl_options = {'REGISTER'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy mesh modellt!")
            return {'CANCELLED'}
        props = context.scene.taf_ml_props
        scale = context.scene.unit_settings.scale_length
        jaw   = props.jaw_type

        present = [LM_CURVE[lm] for lm in LANDMARKS
                   if bpy.data.objects.get(LM_CURVE[lm]) and
                      bpy.data.objects[LM_CURVE[lm]].type == 'CURVE']
        if not present:
            self.report({'ERROR'},
                f"Nem találok berajzolt görbét. Szükséges legalább: {CREST_NAME}.")
            return {'CANCELLED'}

        cast_id = (bpy.path.basename(bpy.data.filepath) or obj.name) + "::" + obj.name
        samples = extract_samples(obj, props, scale, cast_id, jaw)
        if not samples:
            self.report({'ERROR'},
                "0 minta készült. Ellenőrizd a görbe-mesh metszést és a +Z orientációt.")
            return {'CANCELLED'}

        data     = load_dataset(props.dataset_path)
        K        = props.feat_samples
        feat_dim = 2 * K + 3
        if data["samples"] and data.get("feat_dim") not in (None, feat_dim):
            self.report({'ERROR'},
                f"A meglévő adat jellemző-dimenziója ({data.get('feat_dim')}) "
                f"eltér a mostanitól ({feat_dim}). Változott a K – ürítsd a készletet.")
            return {'CANCELLED'}
        data["samples"] = [s for s in data["samples"] if s.get("cast") != cast_id]
        data["samples"].extend(samples)
        data["feat_dim"] = feat_dim
        data["K"]        = K
        path  = save_dataset(props.dataset_path, data)
        ncasts = len({s["cast"] for s in data["samples"]})
        self.report({'INFO'},
            f"+{len(samples)} minta ({jaw}, görbék: {', '.join(present)}). "
            f"Összesen {len(data['samples'])} minta / {ncasts} modell. → {os.path.basename(path)}")
        return {'FINISHED'}


class TAF_ML_OT_Train(Operator):
    """Betanítja a regressziós modellt a teljes tanító készletből és elmenti."""
    bl_idname  = "taf_ml.train"
    bl_label   = "Modell tanítása"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.taf_ml_props
        data  = load_dataset(props.dataset_path)
        if len(data["samples"]) < 8:
            self.report({'ERROR'},
                f"Túl kevés minta ({len(data['samples'])}). Adj hozzá több modellt.")
            return {'CANCELLED'}
        models, counts = train_all(data, props.ridge_lambda)
        if not models:
            self.report({'ERROR'},
                "Egyetlen (állcsont, landmark) párhoz sincs elég minta (min. 8).")
            return {'CANCELLED'}
        out = {"K": data.get("K"), "feat_dim": data.get("feat_dim"),
               "lambda": props.ridge_lambda, "models": models}
        p = _abspath(props.model_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f)
        trained = ", ".join(f"{j}/{lm}:{n}" for (j, lm), n in sorted(counts.items()))
        self.report({'INFO'}, f"Modell mentve ({os.path.basename(p)}). Betanítva: {trained}")
        return {'FINISHED'}


class TAF_ML_OT_Eval(Operator):
    """Kihagyásos (leave-one-cast-out) hibabecslés mm-ben."""
    bl_idname  = "taf_ml.eval"
    bl_label   = "Pontosság (LOO)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props  = context.scene.taf_ml_props
        data   = load_dataset(props.dataset_path)
        ncasts = len({s["cast"] for s in data["samples"]})
        if ncasts < 3:
            self.report({'ERROR'}, f"Legalább 3 modell kell a LOO-hoz (most {ncasts}).")
            return {'CANCELLED'}
        res = loo_eval(data, props.ridge_lambda)
        if not res:
            self.report({'WARNING'}, "Nincs elég adat a kiértékeléshez.")
            return {'CANCELLED'}
        parts = [f"{j}/{lm}: {mae:.2f}mm (n={n})" for (j, lm), (mae, n) in sorted(res.items())]
        self.report({'INFO'}, "LOO átlagos hiba │ " + "  ".join(parts))
        return {'FINISHED'}


class TAF_ML_OT_Clear(Operator):
    """Üríti a tanító készletet (a JSON adatfájlt)."""
    bl_idname  = "taf_ml.clear"
    bl_label   = "Tanító készlet ürítése"
    bl_options = {'REGISTER'}

    def execute(self, context):
        save_dataset(context.scene.taf_ml_props.dataset_path,
                     {"feat_dim": None, "K": None, "samples": []})
        self.report({'INFO'}, "Tanító készlet kiürítve.")
        return {'FINISHED'}


class TAF_ML_OT_Predict(Operator):
    """A betanított modellel legenerálja a görbéket a kijelölt mesh-en."""
    bl_idname  = "taf_ml.predict"
    bl_label   = "Predikció (görbék)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Jelölj ki egy mesh modellt!")
            return {'CANCELLED'}
        props = context.scene.taf_ml_props
        scale = context.scene.unit_settings.scale_length
        jaw   = props.jaw_type

        p = _abspath(props.model_path)
        if not (p and os.path.isfile(p)):
            self.report({'ERROR'}, "Nincs betanított modell. Előbb tanítsd be.")
            return {'CANCELLED'}
        with open(p, "r", encoding="utf-8") as f:
            mdl_file = json.load(f)
        if mdl_file.get("K") not in (None, props.feat_samples):
            self.report({'ERROR'},
                f"A modell K={mdl_file.get('K')} értékkel készült, most K={props.feat_samples}. "
                "Állítsd egyezőre a Profil mintavételt.")
            return {'CANCELLED'}
        jaw_models = (mdl_file.get("models") or {}).get(jaw)
        if not jaw_models:
            self.report({'ERROR'}, f"Ehhez az állcsonthoz ({jaw}) nincs betanított modell.")
            return {'CANCELLED'}

        frames, _ = get_slice_frames(obj, props)
        snap  = mm_to_internal(props.snap_mm, scale)
        seqs  = {lm: [] for lm in jaw_models.keys()}

        for fr in frames:
            pts  = slice_mesh_points(obj, fr.co, fr.normal)
            prof = _to_profile_ml(pts, fr)
            fp   = build_features(prof, props.feat_samples, scale)
            if fp is None:
                continue
            feat, half_width, _, _ = fp
            for lm, mdl in jaw_models.items():
                L_pred = ridge_predict(mdl, feat) * half_width
                seqs[lm].append(_snap_landmark(prof, L_pred, lm, snap))

        if "crest" not in seqs or len(seqs["crest"]) < 4:
            self.report({'ERROR'},
                "Kevés gerincél pont. Ellenőrizd az ívirányt és a +Z orientációt.")
            return {'CANCELLED'}

        s   = props.smooth_strength
        res = props.curve_resolution
        sm  = props.smooth_curve
        made = []
        for lm in LANDMARKS:
            if lm not in seqs or len(seqs[lm]) < 4:
                continue
            pts  = _smooth_sequence(seqs[lm], s)
            cobj = make_nurbs_curve(LM_CURVE[lm], pts, res, sm)
            cobj.data.bevel_depth = 0.0002
            _assign_material(cobj, *LM_COLORS[lm])
            made.append(LM_CURVE[lm])

        self.report({'INFO'}, f"Predikció kész ({jaw}). Görbék: {', '.join(made)}")
        return {'FINISHED'}


# ── Curve Learner panel ───────────────────────────────────────────────────────

class TAF_ML_PT_Panel(Panel):
    bl_label       = "Görbe Tanuló"
    bl_idname      = "TAF_ML_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'TAF'
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props  = context.scene.taf_ml_props

        b0 = layout.box()
        b0.label(text="Állcsont típus", icon='BONE_DATA')
        b0.prop(props, "jaw_type", expand=True)
        b0.label(text="Okkluzális irány = +Z", icon='ORIENTATION_NORMAL')

        b1 = layout.box()
        b1.label(text="Szeletelés (tanítás = predikció!)", icon='SETTINGS')
        c = b1.column(align=True)
        c.prop(props, "arch_axis",  text="Ívirány")
        c.prop(props, "n_slices",   text="Szeletek")
        c.prop(props, "margin_pct", text="Határ levágás")

        b_init = layout.box()
        b_init.label(text="0. Annotáció – görbe iniciátor", icon='GREASEPENCIL')
        b_init.label(text="Jelöld ki a mesh-t, majd:", icon='RESTRICT_SELECT_OFF')
        rinit = b_init.row(align=True)
        rinit.operator("taf_ml.init_curve", text="Gerincél",
                       icon='CURVE_PATH').landmark = 'crest'
        rinit.operator("taf_ml.init_curve", text="Bukkális",
                       icon='CURVE_PATH').landmark = 'buccal'
        if props.jaw_type == 'LOWER':
            b_init.operator("taf_ml.init_curve", text="Linguális",
                            icon='CURVE_PATH').landmark = 'lingual'
        ri = b_init.row(align=True)
        ri.prop(props, "init_surface_draw", text="Felületre rajzolás")
        ri.prop(props, "init_offset_mm",    text="Offset (mm)")

        b2 = layout.box()
        b2.label(text="1. Tanító készlet", icon='IMPORT')
        b2.prop(props, "dataset_path", text="")
        b2.column(align=True).operator("taf_ml.add_sample", icon='ADD')
        row = b2.row(align=True)
        row.operator("taf_ml.eval",  icon='CHECKMARK')
        row.operator("taf_ml.clear", icon='TRASH')

        b3 = layout.box()
        b3.label(text="2. Tanítás", icon='MODIFIER')
        b3.prop(props, "model_path",    text="")
        b3.prop(props, "ridge_lambda",  text="Regularizáció (λ)")
        b3.column(align=True).operator("taf_ml.train", icon='FILE_TICK')

        b4 = layout.box()
        b4.label(text="3. Predikció új modellen", icon='MOD_MESHDEFORM')
        b4.prop(props, "snap_mm", text="Felület-snap (mm)")
        b4.column(align=True).operator("taf_ml.predict", icon='CURVE_DATA')

        b5 = layout.box()
        b5.prop(props, "show_advanced",
                text="Haladó beállítások",
                icon='TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT', emboss=False)
        if props.show_advanced:
            a = b5.column(align=True)
            a.prop(props, "feat_samples",     text="Profil mintavétel (K)")
            a.prop(props, "use_ridge_region", text="Ívillesztés gerincrégióból")
            if props.use_ridge_region:
                a.prop(props, "ridge_region_pct", text="Gerincrégió (felső Z %)")
            a.prop(props, "smooth_strength", text="Pontsor simítás")
            r = a.row(align=True)
            r.prop(props, "smooth_curve",     text="NURBS simítás")
            r.prop(props, "curve_resolution", text="Felbontás")

        s = layout.box()
        s.label(text="Görbe-nevek (annotációhoz):", icon='INFO')
        s.label(text=CREST_NAME)
        s.label(text=BUCCAL_NAME)
        s.label(text=LINGUAL_NAME + "  (alsó)")


# ===========================================================================
# Registration
# ===========================================================================

CLASSES = (
    # Morphometria
    TAF_AddonPreferences,
    TAF_Props,
    TAF_OT_FillPatientId,
    TAF_OT_CalcF1,
    TAF_OT_CalcA2,
    TAF_OT_CalcF2,
    TAF_OT_CalcF3,
    TAF_OT_ExportCSV,
    TAF_OT_ClearFields,
    TAF_OT_AutoF3,
    TAF_OT_PlaceAngleMarker,
    TAF_OT_CalcF6,
    TAF_OT_CalcA10,
    TAF_OT_UploadToServer,
    TAF_OT_SaveAndUploadBlend,
    TAF_OT_AddCursorVertexProbe,
    TAF_PT_Main,
    # Curve Detector
    TAF_CD_Props,
    TAF_OT_DetectCurves,
    TAF_OT_PreviewAxis,
    TAF_OT_ShowArchAxis,
    TAF_PT_CurveDetector,
    # Curve Learner
    TAF_ML_Props,
    TAF_ML_OT_InitCurve,
    TAF_ML_OT_AddSample,
    TAF_ML_OT_Train,
    TAF_ML_OT_Eval,
    TAF_ML_OT_Clear,
    TAF_ML_OT_Predict,
    TAF_ML_PT_Panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taf_props    = bpy.props.PointerProperty(type=TAF_Props)
    bpy.types.Scene.taf_cd_props = bpy.props.PointerProperty(type=TAF_CD_Props)
    bpy.types.Scene.taf_ml_props = bpy.props.PointerProperty(type=TAF_ML_Props)
    if _purge_legacy_scene_credentials not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_purge_legacy_scene_credentials)
    _purge_legacy_scene_credentials()


def unregister():
    if _purge_legacy_scene_credentials in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_purge_legacy_scene_credentials)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.taf_props
    del bpy.types.Scene.taf_cd_props
    del bpy.types.Scene.taf_ml_props


if __name__ == "__main__":
    register()
