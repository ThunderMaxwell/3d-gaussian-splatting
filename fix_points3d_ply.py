#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # SH constant used in many 3DGS codebases

def pick(v, *names):
    for n in names:
        if n in v.dtype.names:
            return v[n]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--assume_rgb_01", action="store_true",
                    help="如果你的 f_dc_* 已经近似是 0~1 的 rgb（很少见），用这个开关绕过 0.5+C0*f_dc")
    args = ap.parse_args()

    inp = Path(args.in_ply)
    out = Path(args.out_ply)

    ply = PlyData.read(str(inp))
    v = ply["vertex"].data
    names = v.dtype.names
    print("[info] vertex props:", names)

    # xyz
    x = np.asarray(pick(v, "x", "X"), dtype=np.float32)
    y = np.asarray(pick(v, "y", "Y"), dtype=np.float32)
    z = np.asarray(pick(v, "z", "Z"), dtype=np.float32)
    if x is None or y is None or z is None:
        raise RuntimeError("缺少 x/y/z")

    npts = len(x)

    # normals (optional; if missing -> 0)
    nx = pick(v, "nx", "normal_x", "NormalX")
    ny = pick(v, "ny", "normal_y", "NormalY")
    nz = pick(v, "nz", "normal_z", "NormalZ")
    if nx is None or ny is None or nz is None:
        nxx = np.zeros(npts, dtype=np.float32)
        nyy = np.zeros(npts, dtype=np.float32)
        nzz = np.zeros(npts, dtype=np.float32)
    else:
        nxx = np.asarray(nx, dtype=np.float32)
        nyy = np.asarray(ny, dtype=np.float32)
        nzz = np.asarray(nz, dtype=np.float32)

    # recover RGB from f_dc_*
    f0 = pick(v, "f_dc_0")
    f1 = pick(v, "f_dc_1")
    f2 = pick(v, "f_dc_2")
    if f0 is None or f1 is None or f2 is None:
        raise RuntimeError("找不到 f_dc_0/1/2，无法恢复颜色")

    fdc = np.stack([np.asarray(f0, dtype=np.float32),
                    np.asarray(f1, dtype=np.float32),
                    np.asarray(f2, dtype=np.float32)], axis=1)

    if args.assume_rgb_01:
        rgb01 = np.clip(fdc, 0.0, 1.0)
    else:
        rgb01 = np.clip(0.5 + C0 * fdc, 0.0, 1.0)

    rgb_u8 = (rgb01 * 255.0 + 0.5).astype(np.uint8)

    # write standard PLY
    dtype = [
        ("x","f4"), ("y","f4"), ("z","f4"),
        ("nx","f4"), ("ny","f4"), ("nz","f4"),
        ("red","u1"), ("green","u1"), ("blue","u1"),
    ]
    arr = np.empty(npts, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = x, y, z
    arr["nx"], arr["ny"], arr["nz"] = nxx, nyy, nzz
    arr["red"], arr["green"], arr["blue"] = rgb_u8[:,0], rgb_u8[:,1], rgb_u8[:,2]

    out.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out))
    print("[ok] wrote:", out)

if __name__ == "__main__":
    main()