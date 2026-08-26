"""
TAF Addon – Blender Addon
=========================
Fogászati morfometriai mérések és biztonságos, verziózott szerveres mentés
egy fájlban.

Modulok
-------
• TAF Morphometria  (N-panel → TAF → TAF Morphometria)

Telepítés
---------
  Edit → Preferences → Add-ons → Install → válaszd ezt a fájlt → engedélyezd

Előfeltétel: a fogászati modell okkluzális iránya = +Z (felfelé).

Szerzői jog: János König, 2024–2025
"""

bl_info = {
    "name": "TAF Addon",
    "author": "János König",
    "version": (2, 3, 0),
    "blender": (3, 0, 0),
    "location": "View3D › Sidebar › TAF",
    "description": "Morfometriai mérések, profil-export és biztonságos szerveres mentés",
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
        import uuid

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

        # Only send values that were actually measured in this Blender file.
        # Property defaults must never erase an earlier server-side measurement.
        payload_dict = {"TAJ": props.patient_id.strip()}
        if props.f1_n_pairs > 0 or props.f1_profile_json:
            payload_dict.update({
                "F1": round(props.f1_upper_ridge_height, 2),
                "F1_ivhossz_mm": (
                    round(props.f1_arc_length_mm, 2) if props.f1_arc_length_mm else None
                ),
                "F1_profil": _parse_profile(props.f1_profile_json),
            })
        if props.f2_original_volume or props.f2_passive_volume or props.f2_undercut_volume:
            payload_dict["F2"] = round(props.f2_undercut_volume, 2)
        if props.f3_palatal_vault:
            payload_dict["F3"] = round(props.f3_palatal_vault, 2)
        if props.f4_angle_left or props.f4_angle_right:
            payload_dict["F4"] = round(f4_avg, 1)
        if props.f6_angle_left or props.f6_angle_right:
            payload_dict["F6"] = round(f6_avg, 1)
        if _A10_FELSO_NAME in bpy.data.objects and _A10_ALSO_NAME in bpy.data.objects:
            payload_dict["A10"] = round(props.a10_angle, 1)
        if props.a2_profile_json or props.a2_lower_ridge_height:
            payload_dict.update({
                "A2_mag_mm": round(props.a2_lower_ridge_height, 2),
                "A2_modszer": props.a2_method,
                "A2_profil": _parse_profile(props.a2_profile_json),
            })
        if props.a2_method_b:
            payload_dict["A2_methodB"] = round(props.a2_method_b, 2)
        if props.a2_method_c:
            payload_dict["A2_methodC"] = round(props.a2_method_c, 2)
        payload = _json.dumps(payload_dict).encode('utf-8')

        # Local backup before uploading
        csv_dir = (os.path.dirname(bpy.path.abspath(props.csv_path))
                   if props.csv_path else "")
        backup_dir = csv_dir or os.path.dirname(bpy.data.filepath or "")
        if not backup_dir or not os.path.isdir(backup_dir):
            self.report({'ERROR'}, "Nincs biztonságos helyi mentési mappa. Mentsd el előbb a .blend fájlt!")
            return {'CANCELLED'}
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = os.path.join(
            backup_dir,
            f"TAF_measurement_{ts}_{uuid.uuid4().hex[:8]}.json"
        )
        try:
            with open(backup_path, 'x', encoding='utf-8') as f:
                _json.dump(payload_dict, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.report({'ERROR'}, f"A helyi biztonsági mentés nem sikerült: {exc}")
            return {'CANCELLED'}

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
                suffix = f" ({attempt}. kísérlet)" if attempt > 1 else ""
                self.report({
                    'INFO'
                }, f"Feltöltve: {props.patient_id}{suffix}; helyi mentés: {os.path.basename(backup_path)}")
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
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.taf_props = bpy.props.PointerProperty(type=TAF_Props)
    if _purge_legacy_scene_credentials not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_purge_legacy_scene_credentials)
    _purge_legacy_scene_credentials()


def unregister():
    if _purge_legacy_scene_credentials in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_purge_legacy_scene_credentials)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.taf_props


if __name__ == "__main__":
    register()
