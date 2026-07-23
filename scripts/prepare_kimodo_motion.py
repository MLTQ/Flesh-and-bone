#!/usr/bin/env python3
"""Retarget one raw Kimodo clip into a periodic H6K LBS motion artifact."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from flesh_and_bone.h4_volume import load_h4_volume
from flesh_and_bone.retarget_skin import canonical_motion_skin, palindrome_close
from flesh_and_bone.rig_asset import linear_skin, load_rig_asset


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archepelago", type=Path, required=True)
    parser.add_argument(
        "--rig", type=Path,
        default=Path("model/derived/meshy_blonde_h4_rig.npz"),
    )
    parser.add_argument(
        "--volume", type=Path,
        default=Path("model/derived/meshy_blonde_h4_volume_p0125.npz"),
    )
    parser.add_argument("--forward-frames", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _destination_rig(asset, Rig):
    heads = asset.rest_bone_endpoints[:, 0].detach().cpu().numpy()
    parents = asset.bone_parents.detach().cpu().numpy()
    offsets = np.empty_like(heads, dtype=np.float32)
    for joint, parent in enumerate(parents):
        offsets[joint] = (
            heads[joint]
            if parent < 0
            else heads[joint] - heads[parent]
        )
    return Rig(
        parents=parents,
        rest_offsets=offsets,
        joint_names=list(asset.bone_names),
    )


def main():
    args = _arguments()
    motion_root = args.archepelago / "backend" / "motion"
    sys.path.insert(0, str(motion_root))
    from skeletor_motion.retarget import (
        build_retarget_map,
        kimodo_npz_to_animation,
        load_soma_rig,
        retarget_animation,
    )
    from skeletor_motion.skeleton import Rig
    from skeletor_motion.skeleton_profile import profile_rig
    from skeletor_motion.transforms import rotation_6d_to_matrix_np

    device = torch.device(args.device)
    cpu_asset = load_rig_asset(args.rig, dtype=torch.float32)
    destination = _destination_rig(cpu_asset, Rig)
    profile = profile_rig(destination)
    soma_path = (
        motion_root / "reports" / "kimodo" / "soma_skeleton.json"
    )
    source = load_soma_rig(soma_path)
    mapping = build_retarget_map(source, destination)
    raw = kimodo_npz_to_animation(args.input)
    retargeted = retarget_animation(mapping, raw)
    local = torch.as_tensor(
        rotation_6d_to_matrix_np(retargeted.rotations_6d),
        device=device,
        dtype=torch.float32,
    )
    root = torch.as_tensor(
        retargeted.root_positions, device=device, dtype=torch.float32
    )
    asset = load_rig_asset(args.rig, device=device, dtype=torch.float32)
    volume = load_h4_volume(args.volume, device=device, dtype=torch.float32)
    skin, endpoints = canonical_motion_skin(asset, local, root)

    identity_local = torch.eye(
        3, device=device, dtype=torch.float32
    ).expand(1, asset.bone_count, 3, 3)
    rest_root = asset.rest_bone_endpoints[
        asset.bone_parents < 0, 0
    ][:1]
    rest_skin, _ = canonical_motion_skin(
        asset, identity_local, rest_root
    )
    identity = torch.eye(4, device=device, dtype=torch.float32)
    rest_identity_error = float((rest_skin - identity).abs().max().item())

    forward = min(args.forward_frames, skin.shape[0])
    lbs_forward = []
    for start in range(0, forward, 5):
        lbs_forward.append(linear_skin(
            volume.points,
            volume.weights,
            skin[start:min(start + 5, forward)],
        ))
    lbs_forward = torch.cat(lbs_forward, dim=0)
    lbs = palindrome_close(lbs_forward, forward).detach().cpu().numpy()
    bones = palindrome_close(endpoints, forward).detach().cpu().numpy()
    root_cycle = palindrome_close(root, forward).detach().cpu().numpy()
    metadata = {
        "format": "flesh-and-bone-h6-kimodo-motion-v1",
        "raw_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "prompt": args.prompt,
        "seed": args.seed,
        "raw_frames": int(local.shape[0]),
        "forward_frames": int(forward),
        "cycle_frames": int(lbs.shape[0]),
        "fps": int(raw.fps),
        "closure": "palindrome-0..N-1,N-2..1",
        "destination_profile": profile.to_dict(),
        "mapped_roles": list(mapping.mapped_roles),
        "mapped_role_count": len(mapping.mapped_roles),
        "root_scale": float(mapping.root_scale),
        "rest_identity_skin_max_error": rest_identity_error,
        "finite": bool(
            np.isfinite(lbs).all() and np.isfinite(bones).all()
        ),
        "root_xz_travel": float(np.linalg.norm(
            root_cycle[:, [0, 2]].max(axis=0)
            - root_cycle[:, [0, 2]].min(axis=0)
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata)),
        lbs_positions=lbs,
        bone_endpoints=bones,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
