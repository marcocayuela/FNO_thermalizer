"""
Downloads a handful of trajectories for ONE (Reynolds, Schmidt) config of
Polymathic's shear_flow dataset (The Well) and reformats them into
thermalizer's own kolmogorov/Re<X>-style convention -- data/shear_flow/
Re<R>_Sc<S>/{train_traj,test_traj}/sim<N>.h5, each holding a "velocity_field"
dataset of shape (T, H, W, 4) (tracer, pressure, u, v) -- so
training.dataset_manager.DatasetManagerMulti (already extended to accept any
shear_flow/ exp_dir, cf. its own comment) and everything built on top of it
(fno_training.py, diffusion_training.py, evaluation/correction_eval.py) work
completely unchanged, just pointed at input_dim=output_dim=4 instead of 2.

Each remote HDF5 (one per (Reynolds, Schmidt), ~13.4 GB for the train split,
32 trajectories bundled with trajectory as the OUTER, non-chunked axis) is
opened LAZILY over HTTP (fsspec) rather than downloaded whole: slicing
[:n_train] on a contiguous (chunks=None) leading axis is a single efficient
byte-range read covering only the requested trajectories, not the full file
-- verified empirically (session notes) before writing this script. Spatial
ds=2 downsampling (256x512 -> 128x256) further keeps the local copy light,
matching the ds convention already used for Kolmogorov/KS.

BCs are periodic in both x and y (cf. the file's own boundary_conditions/
{x,y}_periodic groups) -- same assumption FNO2D's plain spectral convolution
already makes, no Euler-style domain-padding needed here.

Usage:
    python prepare_shear_flow_dataset.py --re 5e4 --schmidt 1e0 \\
        --out_dir /path/to/thermalizer/data --n_train 8 --n_test 4 --ds 2
"""

import argparse
import os

import fsspec
import h5py
import numpy as np
from huggingface_hub import hf_hub_url

REPO_ID = "polymathic-ai/shear_flow"


def _remote_file(split, re_tag, schmidt_tag):
    filename = f"data/{split}/shear_flow_Reynolds_{re_tag}_Schmidt_{schmidt_tag}.hdf5"
    url = hf_hub_url(repo_id=REPO_ID, repo_type="dataset", filename=filename)
    return fsspec.open(url, "rb")


def extract_trajectories(split, re_tag, schmidt_tag, n, ds):
    """Returns a (n, T, H, W, 4) float32 array: tracer, pressure, u, v --
    downloads only the byte range covering the first n trajectories (cf.
    module docstring)."""
    with _remote_file(split, re_tag, schmidt_tag) as f:
        with h5py.File(f, "r") as hf:
            tracer = hf["t0_fields/tracer"][:n, :, ::ds, ::ds]
            pressure = hf["t0_fields/pressure"][:n, :, ::ds, ::ds]
            velocity = hf["t1_fields/velocity"][:n, :, ::ds, ::ds, :]  # (n, T, H, W, 2)
    return np.stack([tracer, pressure], axis=-1).astype(np.float32), velocity.astype(np.float32)
    # (n,T,H,W,2) tracer+pressure, (n,T,H,W,2) u,v -- concatenated by the caller


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--re", required=True, help="Reynolds tag exactly as in the HF filename, e.g. 5e4")
    parser.add_argument("--schmidt", required=True, help="Schmidt tag exactly as in the HF filename, e.g. 1e0")
    parser.add_argument("--out_dir", required=True, help="thermalizer data/ root")
    parser.add_argument("--n_train", type=int, default=8, help="<=32 available in the train file")
    parser.add_argument("--n_test", type=int, default=4, help="<=4 available in the test file")
    parser.add_argument("--ds", type=int, default=2, help="Spatial downsample stride (256x512 -> 128x256 at ds=2)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tag = f"Re{args.re}_Sc{args.schmidt}"
    root = os.path.join(args.out_dir, "shear_flow", tag)

    for split, out_split, n in (("train", "train_traj", args.n_train), ("test", "test_traj", args.n_test)):
        split_dir = os.path.join(root, out_split)
        os.makedirs(split_dir, exist_ok=True)

        print(f"- fetching {n} trajectories from {split} (nu={args.re}, Schmidt={args.schmidt})...", flush=True)
        scalars, velocity = extract_trajectories(split, args.re, args.schmidt, n, args.ds)
        combined = np.concatenate([scalars, velocity], axis=-1)  # (n, T, H, W, 4): tracer, pressure, u, v

        for i in range(n):
            out_path = os.path.join(split_dir, f"sim{i + 1}.h5")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"  {out_path} already exists, skipping (--overwrite to force)", flush=True)
                continue
            with h5py.File(out_path, "w") as f:
                f.create_dataset("velocity_field", data=combined[i])
                f.attrs["reynolds"] = args.re
                f.attrs["schmidt"] = args.schmidt
                f.attrs["field_order"] = "tracer,pressure,u,v"
                f.attrs["ds"] = args.ds
            print(f"  saved {combined[i].shape} to {out_path}", flush=True)

    print(f"Done. Dataset at {root}/", flush=True)


if __name__ == "__main__":
    main()
