"""Turntable-aware SfM refinement: fit the camera rig, refine it with a small
bundle adjustment, and re-triangulate the input tracks against the fixed poses.
Falls back to masked SIFT when the input has no usable tracks."""
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Optional

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as Rot

from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.base import SceneRefiner, SfMResult

logger = logging.getLogger(__name__)


def group_key(name, regex):
    m = re.search(regex, os.path.basename(name))
    return m.group(0) if m else "cam"


def order_key(name):
    nums = re.findall(r'\d+', os.path.basename(name))
    return int(nums[-1]) if nums else 0


def Rt(image):
    T = image.cam_from_world().matrix()
    return T[:, :3], T[:, 3]


def centers(ims):
    return np.array([-(Rt(im)[0].T @ Rt(im)[1]) for im in ims])


def Rf(axis, theta_deg):
    return Rot.from_rotvec(np.radians(theta_deg) * axis).as_matrix()


def fit_axis_step(groups):
    normals, mids = [], []
    for ims in groups.values():
        C = centers(ims)
        c = C.mean(0)
        _, _, Vt = np.linalg.svd(C - c)
        n = Vt[2]
        n = -n if n[1] < 0 else n
        normals.append(n)
        mids.append(c)
    axis = np.mean(normals, 0)
    axis /= np.linalg.norm(axis)
    p0 = np.mean(mids, 0)

    # angular step: least-squares slope of unwrapped angle vs frame index
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(e1 @ axis) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    slopes = []
    for ims in groups.values():
        if len(ims) < 3:
            continue
        v = centers(ims) - p0
        v = v - np.outer(v @ axis, axis)
        ang = np.unwrap(np.arctan2(v @ e2, v @ e1))
        idx = np.arange(len(ang))
        slope = np.polyfit(idx, ang, 1)[0]
        slopes.append(np.degrees(slope))
    return axis, p0, float(np.median(slopes))


def fit_rig_poses(groups, axis, p0, step):
    out, dc, dr, rig = {}, [], [], {}
    for k, ims in groups.items():
        Rc_est, Cc_est = [], []
        for i, im in enumerate(ims):
            R, t = Rt(im)
            C = -R.T @ t
            Rc_est.append(R @ Rf(axis, i * step).T)
            Cc_est.append(p0 + Rf(axis, i * step) @ (C - p0))
        Rc = Rot.from_matrix(np.array(Rc_est)).mean().as_matrix()
        Cc = np.mean(Cc_est, 0)
        rig[k] = (Rc, Cc)
        for i, im in enumerate(ims):
            Rn = Rc @ Rf(axis, i * step)
            Cn = p0 + Rf(axis, i * step).T @ (Cc - p0)
            R, t = Rt(im)
            dc.append(np.linalg.norm(Cn - (-R.T @ t)))
            dr.append(np.degrees(np.linalg.norm(Rot.from_matrix(Rn @ R.T).as_rotvec())))
            out[im.name] = (Rn, Cn)
    return out, float(np.mean(dc)), float(np.mean(dr)), rig


def _batch_dlt(P, uv):
    x, y = uv[..., 0], uv[..., 1]
    A = np.concatenate([x[..., None] * P[..., 2, :] - P[..., 0, :],
                        y[..., None] * P[..., 2, :] - P[..., 1, :]], axis=1)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1, :]
    return Xh[:, :3] / Xh[:, 3:4]


def apply_track_preserving(rec, poses, regex, max_reproj=None, min_track=3, max_obs=20):
    """Snap each image to its rig pose and re-triangulate the existing tracks,
    dropping points over max_reproj (default 2.5x the median). Mutates rec."""
    ids = sorted(rec.images.keys())
    ipos = {iid: i for i, iid in enumerate(ids)}
    N = len(ids)
    Pn = np.zeros((N, 3, 4))
    intr = np.zeros((N, 3))
    xy = {}
    for iid in ids:
        im = rec.images[iid]
        R, C = poses[im.name]
        Pn[ipos[iid]] = np.hstack([R, (-R @ C).reshape(3, 1)])
        intr[ipos[iid]] = rec.cameras[im.camera_id].params[:3]
        xy[ipos[iid]] = np.array([p.xy for p in im.points2D])

    by_len = defaultdict(lambda: ([], [], []))   # L -> (pids, img-idx rows, uv rows)
    to_delete = []
    for pid, pt in rec.points3D.items():
        els = pt.track.elements
        if len(els) < min_track:
            to_delete.append(pid)
            continue
        if len(els) > max_obs:
            els = els[:max_obs]
        ii = np.fromiter((ipos[e.image_id] for e in els), int, len(els))
        uvv = np.array([xy[i][e.point2D_idx] for e, i in zip(els, ii)])
        by_len[len(ii)][0].append(pid)
        by_len[len(ii)][1].append(ii)
        by_len[len(ii)][2].append(uvv)

    cand = {}
    for L, (pids, iis, uvs) in by_len.items():
        ii = np.stack(iis); uv = np.stack(uvs)
        f = intr[ii, 0]; cx = intr[ii, 1]; cy = intr[ii, 2]
        uvn = np.stack([(uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f], axis=-1)
        Pm = Pn[ii]
        X = _batch_dlt(Pm, uvn)
        Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
        Xc = np.einsum('mlij,mj->mli', Pm, Xh)
        uu = f * Xc[..., 0] / Xc[..., 2] + cx
        vv = f * Xc[..., 1] / Xc[..., 2] + cy
        err = np.hypot(uu - uv[..., 0], vv - uv[..., 1]).mean(1)
        front = (Xc[..., 2] > 0).all(1)
        for k, pid in enumerate(pids):
            cand[pid] = (X[k], float(err[k]), bool(front[k]), L)

    if max_reproj is None:
        med = np.median([c[1] for c in cand.values()]) if cand else 1.0
        max_reproj = max(2.0, 2.5 * med)

    errs, tls = [], []
    for pid, (X, err, front, L) in cand.items():
        if err < max_reproj and front:
            rec.points3D[pid].xyz = X.tolist()
            errs.append(err); tls.append(L)
        else:
            to_delete.append(pid)

    for iid in ids:
        im = rec.images[iid]
        R, C = poses[im.name]
        q = Rot.from_matrix(R).as_quat()
        im.frame.set_cam_from_world(
            im.camera_id,
            pycolmap.Rigid3d(pycolmap.Rotation3d([q[0], q[1], q[2], q[3]]), -R @ C))
    for pid in to_delete:
        rec.delete_point3D(pid)

    return dict(pts=len(errs), thr=float(max_reproj),
                track=float(np.mean(tls)) if tls else 0.0,
                reproj=float(np.mean(errs)) if errs else -1.0)


def rig_poses(groups, axis, p0, step, rig):
    axis = axis / np.linalg.norm(axis)
    out = {}
    for k, ims in groups.items():
        Rc, Cc = rig[k]
        for i, im in enumerate(ims):
            Ri = Rf(axis, i * step)
            out[im.name] = (Rc @ Ri, p0 + Ri.T @ (Cc - p0))
    return out


def _triangulate_batched(Rn, Cn, f, cx, cy, uv, offs, cnts):
    tn = -np.einsum('mij,mj->mi', Rn, Cn)
    P = np.concatenate([Rn, tn[:, :, None]], axis=2)          # (M,3,4)
    uvn = np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f], axis=-1)
    npts = len(offs)
    X = np.zeros((npts, 3))
    err = np.zeros(npts)
    order = np.argsort(cnts)
    L = cnts[order]
    for lval in np.unique(L):
        rows = order[L == lval]
        idx = np.stack([np.arange(offs[r], offs[r] + lval) for r in rows])  # (m,lval)
        Pm = P[idx]                                            # (m,lval,3,4)
        uvm = uvn[idx]                                         # (m,lval,2)
        Xm = _batch_dlt(Pm, uvm)
        Xh = np.concatenate([Xm, np.ones((len(Xm), 1))], 1)
        Xc = np.einsum('mlij,mj->mli', Pm, Xh)
        z = np.where(np.abs(Xc[..., 2]) < 1e-9, 1e-9, Xc[..., 2])
        uu = f[idx] * Xc[..., 0] / z + cx[idx]
        vv = f[idx] * Xc[..., 1] / z + cy[idx]
        err[rows] = np.hypot(uu - uv[idx][..., 0], vv - uv[idx][..., 1]).mean(1)
        X[rows] = Xm
    return X, err


def rig_ba(rec, groups, axis, p0, step, rig, min_track=3, max_obs=20,
           sample=60000, iters=3, refine_focal=False):
    """Refine the rig (axis, step, centre, per-camera pose) by alternating
    point re-triangulation with an LM fit against reprojection error."""
    from scipy.optimize import least_squares

    gk = sorted(rig.keys())
    gi = {k: i for i, k in enumerate(gk)}
    G = len(gk)

    img_g, img_i, img_f, img_c = {}, {}, {}, {}
    for k in gk:
        for i, im in enumerate(groups[k]):
            cam = rec.cameras[im.camera_id]
            img_g[im.image_id] = gi[k]
            img_i[im.image_id] = i
            img_f[im.image_id] = float(cam.params[0])
            img_c[im.image_id] = np.asarray(cam.params[1:3], float)
    xy = {im.image_id: np.array([q.xy for q in im.points2D])
          for im in rec.images.values()}

    offs, cnts, X0 = [], [], []
    obs_pt, obs_g, obs_i, obs_uv, obs_f, obs_c = [], [], [], [], [], []
    cur = 0
    for pt in rec.points3D.values():
        els = pt.track.elements
        if len(els) < min_track:
            continue
        if len(els) > max_obs:
            els = els[:max_obs]
        prow = len(offs)
        offs.append(cur); cnts.append(len(els)); X0.append(list(pt.xyz))
        for e in els:
            iid = e.image_id
            obs_pt.append(prow); obs_g.append(img_g[iid]); obs_i.append(img_i[iid])
            obs_uv.append(xy[iid][e.point2D_idx])
            obs_f.append(img_f[iid]); obs_c.append(img_c[iid])
        cur += len(els)
    if not offs:
        return axis, p0, step, rig, dict(pts=0, reproj0=-1.0, reproj=-1.0)
    offs = np.array(offs); cnts = np.array(cnts)
    obs_pt = np.array(obs_pt); obs_g = np.array(obs_g)
    obs_i = np.array(obs_i, float); obs_uv = np.array(obs_uv)
    obs_f = np.array(obs_f); obs_c = np.array(obs_c)
    X = np.array(X0, float)
    M = len(obs_pt)

    rng = np.random.default_rng(0)
    sel = rng.choice(M, sample, replace=False) if M > sample else np.arange(M)
    s_pt, s_g, s_i = obs_pt[sel], obs_g[sel], obs_i[sel]
    s_uv, s_f, s_c = obs_uv[sel], obs_f[sel], obs_c[sel]

    rc0 = np.array([Rot.from_matrix(rig[k][0]).as_rotvec() for k in gk])
    cc0 = np.array([rig[k][1] for k in gk])

    def unpack(prm):
        ax = prm[0:3]; ax = ax / (np.linalg.norm(ax) + 1e-12)
        p0v = prm[3:6]; st = prm[6]
        rc = prm[7:7 + 3 * G].reshape(G, 3)
        cc = prm[7 + 3 * G:7 + 6 * G].reshape(G, 3)
        fsc = prm[7 + 6 * G:] if refine_focal else np.ones(G)
        return ax, p0v, st, rc, cc, fsc

    def poses_for(prm, gidx, iidx):
        ax, p0v, st, rc, cc, fsc = unpack(prm)
        Ri = Rot.from_rotvec(np.radians(iidx * st)[:, None] * ax[None, :]).as_matrix()
        Rc = Rot.from_rotvec(rc).as_matrix()[gidx]
        Rn = np.einsum('mij,mjk->mik', Rc, Ri)
        Cn = p0v + np.einsum('mji,mj->mi', Ri, cc[gidx] - p0v)
        return Rn, Cn, fsc

    def resid(prm):
        Rn, Cn, fsc = poses_for(prm, s_g, s_i)
        Xc = np.einsum('mij,mj->mi', Rn, X[s_pt] - Cn)
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        fe = s_f * fsc[s_g]
        u = fe * Xc[:, 0] / z + s_c[:, 0]
        v = fe * Xc[:, 1] / z + s_c[:, 1]
        return np.concatenate([u - s_uv[:, 0], v - s_uv[:, 1]])

    prm = np.concatenate([axis / np.linalg.norm(axis), p0, [step],
                          rc0.ravel(), cc0.ravel()])
    if refine_focal:
        prm = np.concatenate([prm, np.ones(G)])

    reproj0 = float(np.abs(resid(prm)).mean())
    for _ in range(iters):
        Rn, Cn, fsc = poses_for(prm, obs_g, obs_i)
        fe = obs_f * fsc[obs_g]
        X, _ = _triangulate_batched(Rn, Cn, fe, obs_c[:, 0], obs_c[:, 1],
                                    obs_uv, offs, cnts)
        prm = least_squares(resid, prm, method='lm', max_nfev=200).x
    reproj = float(np.abs(resid(prm)).mean())

    ax, p0v, st, rc, cc, fsc = unpack(prm)
    rig_out = {k: (Rot.from_rotvec(rc[gi[k]]).as_matrix(), cc[gi[k]]) for k in gk}
    stats = dict(pts=len(offs), reproj0=reproj0, reproj=reproj,
                 step0=float(step), step=float(st))
    return ax, p0v, float(st), rig_out, stats


@dataclass(frozen=True)
class TurntableConfig:
    """Rig-fitting, retriangulation, and fallback-SIFT parameters."""

    use_masks: bool = field(default=False, metadata={
        "help": "Restrict SIFT to masks/ (auto-on if masks/ exists)"})
    camera_regex: str = field(default=r"camera\d+", metadata={
        "help": "Regex grouping images into physical cameras"})
    step_deg: Optional[float] = field(default=None, metadata={
        "help": "Override the turntable angular step in degrees"})
    max_image_size: int = field(default=2400, metadata={"help": "Max image dimension for SIFT"})
    retriangulate: Literal["auto", "tracks", "sift"] = field(default="auto", metadata={
        "help": "Point source: reuse input tracks ('tracks'), re-run masked "
                "SIFT ('sift'), or auto-pick by input track length ('auto')"})
    max_reproj: Optional[float] = field(default=None, metadata={
        "help": "Max mean reprojection error (px) to keep a track; "
                "default adapts to 2.5x the median"})
    rig_ba: Literal["auto", "on", "off"] = field(default="auto", metadata={
        "help": "Rig-constrained bundle adjustment before re-triangulation; "
                "'auto'/'on' enable it, 'off' skips it for ablation"})
    rig_ba_iters: int = field(default=3, metadata={
        "help": "Resection-intersection rounds for rig BA"})


@register_sfm
class TurntableRefiner(SceneRefiner):
    """Refines an existing COLMAP scene by fitting and enforcing a turntable rig."""

    name: ClassVar[str] = "turntable"
    config_cls: ClassVar[type] = TurntableConfig

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Fit the rig, then either re-triangulate the input tracks or re-run masked SIFT.

        Args:
            scene: Existing COLMAP scene to refine.
            output_dir: Directory to write the refined COLMAP scene into.

        Returns:
            An SfMResult for the refined reconstruction.
        """
        self.validate(scene)
        args = self.config

        t_start = time.time()
        in_dir = scene.root.resolve()
        out_dir = output_dir.resolve()
        images_dir = in_dir / "images"
        masks_dir = in_dir / "masks"
        use_masks = args.use_masks or masks_dir.is_dir()

        rec = pycolmap.Reconstruction(str(in_dir / "sparse" / "0"))
        groups = defaultdict(list)
        for im in rec.images.values():
            groups[group_key(im.name, args.camera_regex)].append(im)
        for k in groups:
            groups[k].sort(key=lambda im: order_key(im.name))
        logger.info("Grouped %d images into %d cameras: %s", rec.num_images(), len(groups),
                    ", ".join(f"{k}:{len(v)}" for k, v in sorted(groups.items())))

        axis, p0, step_meas = fit_axis_step(groups)
        step = args.step_deg if args.step_deg is not None else step_meas
        candidates = [step] if args.step_deg is not None else [step, -step]

        # step sign is ambiguous; keep whichever fits the input centers best
        poses = best = rig = None
        for s in candidates:
            ps, dc, dr, rg = fit_rig_poses(groups, axis, p0, s)
            if best is None or dc < best[0]:
                best, poses, step, rig = (dc, dr), ps, s, rg
        dc, dr = best
        logger.info("Rig axis %s, step %.3f deg; analytic vs input: dC %.4f, dR %.3f deg",
                    np.round(axis, 4), step, dc, dr)

        # reuse dense tracks if present, else fall back to masked SIFT
        in_track = np.mean([pt.track.length() for pt in rec.points3D.values()]) \
            if rec.num_points3D() else 0.0
        mode = args.retriangulate
        if mode == "auto":
            mode = "tracks" if in_track >= 3.0 else "sift"
        logger.info("Input mean track length %.1f -> retriangulation mode: %s", in_track, mode)

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        if not (out_dir / "images").exists():
            os.symlink(images_dir, out_dir / "images")

        if mode == "tracks":
            return self._run_tracks(rec, groups, poses, axis, p0, step, rig,
                                    scene, out_dir, masks_dir, use_masks, t_start)
        return self._run_sift(rec, poses, scene, out_dir, images_dir, masks_dir,
                              use_masks, t_start)

    def _run_tracks(self, rec, groups, poses, axis, p0, step, rig,
                    scene, out_dir, masks_dir, use_masks, t_start) -> SfMResult:
        """Snap poses to the refined rig and re-triangulate the input tracks."""
        args = self.config
        out_sparse = str(out_dir / "sparse" / "0")
        if use_masks and not (out_dir / "masks").exists():
            os.symlink(masks_dir, out_dir / "masks")

        do_ba = args.rig_ba != "off"
        if do_ba:
            axis, p0, step, rig, bs = rig_ba(rec, groups, axis, p0, step, rig,
                                             iters=args.rig_ba_iters)
            poses = rig_poses(groups, axis, p0, step, rig)
            logger.info("Rig BA: %d pts, step %.3f->%.3f deg, reproj %.2f->%.2f px",
                        bs["pts"], bs["step0"], bs["step"], bs["reproj0"], bs["reproj"])

        stats = apply_track_preserving(rec, poses, args.camera_regex,
                                       max_reproj=args.max_reproj)

        for cid, cam in list(rec.cameras.items()):
            if "SIMPLE_PINHOLE" not in str(cam.model):
                f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
                new = pycolmap.Camera.create(cid, "SIMPLE_PINHOLE", float(f),
                                             cam.width, cam.height)
                new.params = [float(f), float(cx), float(cy)]
                rec.cameras[cid] = new
        rec.write(out_sparse)
        logger.info("Track-preserving: %d points, mean track %.1f, reproj %.2f px (thr %.1f)",
                    stats["pts"], stats["track"], stats["reproj"], stats["thr"])
        elapsed = time.time() - t_start
        logger.info("Done in %.1fs -> %s", elapsed, out_dir)
        return SfMResult(output_dir=out_dir, elapsed=elapsed, scene=Scene(out_dir),
                         num_images=rec.num_reg_images(), num_points=rec.num_points3D())

    def _run_sift(self, rec, poses, scene, out_dir, images_dir, masks_dir,
                  use_masks, t_start) -> SfMResult:
        """Re-run masked SIFT and triangulate against the fixed rig poses."""
        args = self.config

        foc = defaultdict(list)
        W = H = None
        for im in rec.images.values():
            cam = rec.cameras[im.camera_id]
            foc[group_key(im.name, args.camera_regex)].append(cam.params[0])
            W, H = cam.width, cam.height
        shared_focal = {k: float(np.median(v)) for k, v in foc.items()}

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        if not (out_dir / "images").exists():
            os.symlink(images_dir, out_dir / "images")

        mask_path = None
        if use_masks:
            if not (out_dir / "masks").exists():
                os.symlink(masks_dir, out_dir / "masks")
            # COLMAP looks for <image_name>.png; images are .jpg, so link <stem>.jpg.png
            mask_dir = scene.link_colmap_masks(out_dir / "masks_colmap")
            mask_path = str(mask_dir) if mask_dir else None

        db_path = str(out_dir / "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        ro = pycolmap.ImageReaderOptions()
        if mask_path:
            ro.mask_path = mask_path
        ro.camera_model = "SIMPLE_PINHOLE"
        eo = pycolmap.FeatureExtractionOptions()
        eo.max_image_size = args.max_image_size
        n_threads = len(os.sched_getaffinity(0))
        eo.num_threads = n_threads
        mo = pycolmap.FeatureMatchingOptions()
        mo.num_threads = n_threads
        po = pycolmap.IncrementalPipelineOptions()
        po.num_threads = n_threads

        t0 = time.time()
        pycolmap.extract_features(db_path, str(out_dir / "images"),
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=ro, extraction_options=eo)
        logger.info("SIFT extraction %.0fs", time.time() - t0)
        t0 = time.time()
        pycolmap.match_exhaustive(db_path, matching_options=mo)
        logger.info("Matching %.0fs", time.time() - t0)

        db = pycolmap.Database()
        db.open(db_path)
        db_images = db.read_all_images()
        db.close()

        out_rec = pycolmap.Reconstruction()
        cam_id = {}
        for ci, k in enumerate(sorted(shared_focal), start=1):
            cam = pycolmap.Camera.create(ci, "SIMPLE_PINHOLE", shared_focal[k], W, H)
            cam.camera_id = ci
            # pycolmap>=4 needs every camera to belong to a rig.
            out_rec.add_camera_with_trivial_rig(cam)
            cam_id[k] = ci
        for dbi in db_images:
            R, C = poses[dbi.name]
            q = Rot.from_matrix(R).as_quat()  # x, y, z, w
            rigid = pycolmap.Rigid3d(pycolmap.Rotation3d([q[0], q[1], q[2], q[3]]), -R @ C)
            img = pycolmap.Image(name=dbi.name,
                                 camera_id=cam_id[group_key(dbi.name, args.camera_regex)],
                                 image_id=dbi.image_id)
            # pycolmap>=4 keeps the pose on the frame; supplying one also registers the image.
            out_rec.add_image_with_trivial_frame(img, rigid)

        out_sparse = str(out_dir / "sparse" / "0")
        tri = pycolmap.triangulate_points(out_rec, db_path, str(out_dir / "images"), out_sparse,
                                          options=po)
        tl = [pt.track.length() for pt in tri.points3D.values()]
        tri.write(out_sparse)
        logger.info("Triangulated %d points, %d images, mean track %.2f",
                    tri.num_points3D(), tri.num_reg_images(), np.mean(tl))
        elapsed = time.time() - t_start
        logger.info("Done in %.1fs -> %s", elapsed, out_dir)
        return SfMResult(output_dir=out_dir, elapsed=elapsed, scene=Scene(out_dir),
                         num_images=tri.num_reg_images(), num_points=tri.num_points3D())
