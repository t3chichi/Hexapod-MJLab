#!/usr/bin/env python3
"""Historical one-off organization of logs/verification (2026-10-19).

Retained as a record of that operation, not as the current cleanup tool.
The original input runs were removed in the 2026-10-27 cleanup; the checks
below intentionally reject today's empty output roots. Original mappings,
validation and operations are preserved and require the original dataset.

Default is READ-ONLY audit. This script never touches logs outside
logs/verification, never deletes/renames checkpoints, historical
feature-cache-20261013, or protected immutable run recipes. FAIL witnesses
from superseded small experiments are copied into retained_failures/ first.

Usage from Hexapod-MJLab root:
  python scripts/tools/organize_verification_logs.py --plan
  python scripts/tools/organize_verification_logs.py --apply

--apply is deliberately explicit. Do not run it during an ongoing
verification job or repoint its historical plan at new experiment data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "logs" / "verification"

# Only externally agreed Stage-A static evidence and earlier CPU summary.
# The old->new mapping is also retained as an audit trail: embedded path
# strings in frozen JSON are historical; their payload bytes are NOT edited.
MOVE = {
    "stageA_preflight_20261014_t1_v1":
        "historical/20261014_initial_scanner_baseline",
    "stageA_B_cone_cutoff_20261015_v1":
        "current/near_pointcloud/01_angular_coverage_20261015",
    "stageA_B_length_20261015_v1":
        "current/near_pointcloud/02_forward_range_20261015",
    "stageA_B8_corridors_20261015_v3":
        "current/near_pointcloud/03_corridor_wall_coverage_20261015",
    "stageA_B8_pose_perturb_20261015_v6":
        "current/near_pointcloud/04_pose_noise_and_scan_gap_20261015",
    "stageA_B8_formal_preflight_20261019_v1":
        "current/near_pointcloud/05_production_scan_20261019",
    "stageA_B8_ground_support_20261019_v4":
        "current/ground_height/01_ground_visibility_by_pose_20261019",
    "stageA_S1_strict_rect_20261019_v1":
        "current/ground_height/02_rectangle_extent_comparison_20261019",
    "stageA_S1_grid_compare_20261019_v1":
        "historical/20261019_preliminary_grid_density_comparison",
    "stageA_S1_187_pose_support_20261019_v1":
        "current/ground_height/03_fixed_grid_187_pose_validation_20261019",
    "confined_cpu-20261019-040130-204224":
        "current/cpu_regression/20261019_14_groups_passed",
    # Preserve this unique failing regression as a NEGATIVE witness.
    "confined_cpu-20261019-035331-120148":
        "historical/20261019_cpu_play_failure_before_fix",
}
# Explicitly superseded diagnostics. We archive FAILED receipts before
# deletion; the successful authority for each experiment is in MOVE.
DROP = (
    "confined_cpu-20261015-200940-799876",
    "confined_cpu-20261019-023148-531342",
    "confined_cpu-20261019-023355-948168",
    "confined_cpu-20261019-024714-268463",
    "confined_cpu-20261019-035542-047297",
    "stageA_B8_corridors_20261015_v1",
    "stageA_B8_corridors_20261015_v2",
    "stageA_B8_pose_perturb_20261015_v1",
    "stageA_B8_pose_perturb_20261015_v2",
    "stageA_B8_pose_perturb_20261015_v3",
    "stageA_B8_pose_perturb_20261015_v4",
    "stageA_B8_pose_perturb_20261015_v5",
    "stageA_B8_gap_independent_20261019_v1",
    "stageA_B8_gap_independent_20261019_v2",
    "stageA_B8_ground_support_20261019_v1",
    "stageA_B8_ground_support_20261019_v2",
)
# Important: stageA_B8_ground_support_20261019_v3 is still the original
# source artifact for the early grid-density comparison and is retained.
# GPU lifecycle/actual checkpoints are kept IN PLACE: their run recipes
# contain absolute paths, and moving them would invalidate exact resume.
PROTECT = (
    "feature-cache-20261013",
    "stageA_B8_lifecycle_gpu_20261019_v1",
    "stageA_B8_production_gpu_20261019_v1",
    "stageA_B8_audit_gpu_20261019_v1",
    "stageA_B8_gap_independent_20261019_v3",
    "stageA_S1_187_formal_gpu_20261019_v1",
    "stageA_B8_ground_support_20261019_v3",
    "nominal-soft-limits-20261005",
)
# These are VIRTUAL navigation links only: do not move real GPU run
# directories with checkpoint files or embedded absolute paths.
GPU_LINKS = {
    "current/gpu_verification/01_scanner_runtime": "stageA_B8_production_gpu_20261019_v1",
    "current/gpu_verification/02_range_noise": "stageA_B8_audit_gpu_20261019_v1",
    "current/gpu_verification/03_scan_gap": "stageA_B8_gap_independent_20261019_v3",
    "current/gpu_verification/04_training_lifecycle": "stageA_B8_lifecycle_gpu_20261019_v1",
    "current/gpu_verification/05_fixed_height_grid_training": "stageA_S1_187_formal_gpu_20261019_v1",
}
# Naming changes for human-oriented figures only, NOT frozen JSON/.pt.
FIGURES = {
    "near_hit_grid_6panels.png": "near_cone_ground_hits_top_view.png",
    "fps_far_6panels.png": "near_fps256_far64_top_view.png",
    "length_sweep_pool_8panels.png": "forward_range_full_hit_comparison.png",
    "length_sweep_fps_8panels.png": "forward_range_fps256_comparison.png",
    "corridors_B8_A0_B9_bev.png": "corridor_wall_sampling_top_view.png",
    "pose_wall_fps_comparison.png": "sidewall_fps_by_robot_pose.png",
    "perturbation_group_changes.png": "range_gap_group_change_by_scan.png",
    "B8_ground_pose_support_6panels.png": "ground_coverage_across_15_poses.png",
    "B8_ground_boundary_comparison.png": "ground_coverage_rectangle_boundaries.png",
    "B8_supervision_S1_S2_overlay.png": "wide_vs_conservative_coverage.png",
    "strict_S1_rectangle_9poses.png": "stable_rectangle_intersection_9_poses.png",
}

# The cleanup must never mass-edit experimental scripts, tests or frozen
# identities. Only the current authority doc and the single analysis script
# whose source DATA PATH gets relocated are updated; exact content changes
# are recorded in the manifest. No .pt or historical .json is rewritten.
SOURCE_ALLOWLIST = (
    "docs/tasks/confined_locomotion/lidar_setup/配置与边界.md",
    "scripts/experiments/confined_locomotion/compare_S1_strict_rectangles.py",
)
RECEIPT_SUFFIXES = {".json", ".log", ".txt"}
INDEX = """# 验证记录导航

本索引按**用途**组织，而不是按实验代号B8/S1重复命名文件夹。B8、S1-187是当前有效配置的版本代号，仅在报告和配方中标明。

## 当前近端点云（B8, FPS256）
- [近端角域与覆盖](current/near_pointcloud/01_angular_coverage_20261015/)
- [前向截断范围比较](current/near_pointcloud/02_forward_range_20261015/)
- [狭窄通道侧壁回波](current/near_pointcloud/03_corridor_wall_coverage_20261015/)
- [多姿态、噪声与缺口](current/near_pointcloud/04_pose_noise_and_scan_gap_20261015/)
- [正式扫描器基线](current/near_pointcloud/05_production_scan_20261019/)

## 当前相对地面高度监督（187点）
- [近端地面可观测性及严格锥域](current/ground_height/01_ground_visibility_by_pose_20261019/)
- [固定矩形边界比较](current/ground_height/02_rectangle_extent_comparison_20261019/)
- [187个位置的15姿态验证](current/ground_height/03_fixed_grid_187_pose_validation_20261019/)
- [CPU14组最新通过](current/cpu_regression/20261019_14_groups_passed/)
- [187点GPU有界训练及完整Checkpoint（原路径保留）](current/gpu_verification/05_fixed_height_grid_training/)

## GPU工程门（这里是快捷链接，原run路径与checkpoint不变）
- [扫描与自然采集](current/gpu_verification/01_scanner_runtime/)
- [消费后测距扰动与源身份](current/gpu_verification/02_range_noise/)
- [真实扫描缺口触发复算](current/gpu_verification/03_scan_gap/)
- [重置、联合PPO更新与完整Checkpoint验证](current/gpu_verification/04_training_lifecycle/)

## 历史研究及失败证据
- [近远端编码缓存最终权威报告](feature-cache-20261013/final_report.json)
- [缓存加速性能对照](feature-cache-20261013/performance_summary.json)
- [RNN/IEEE数值误差与失败分析](feature-cache-20261013/gpu-cache-diagnostic-rnn/diagnostic.json)
- [早期初始雷达扫描](historical/20261014_initial_scanner_baseline/)
- [早期网格数量对照](historical/20261019_preliminary_grid_density_comparison/)
- [2026-10-19 CPU Play故障原件](historical/20261019_cpu_play_failure_before_fix/)
- [早期密集探测网格v3（早期密度对照的依赖）](stageA_B8_ground_support_20261019_v3/)
- [其他任务软限验证](nominal-soft-limits-20261005/)
- [失败版本保留的报告、配方与日志](retained_failures/)

## 目录规则与审计
- `current/` 内以研究内容命名，不以B8或S1作目录前缀。以`01_`排序实验过程，不表示模型版本。
- `historical/` 是有独立历史价值但不是当前配置的原始证据。
- `feature-cache-20261013/` 是较早独立阶段的历史验收集合，核心GPU大目录在无进一步单独授权时保留。
- 含Checkpoint和绝对运行路径的GPU目录不移动。为了方便浏览，`current/gpu_verification/`放的是Linux**相对符号链接**，只导航不复制或修改数据。所有保留的`recipe.json`、`report.json`及`.pt`保持原始字节和身份；被删除的失败实验另存小型文本原件。报告内部旧路径通过[移动与删除记录](verification_organize_manifest.json)查证。
- 被淘汰的失败试验先留存小型文本证据，才清除中间数组；新实验必须使用新输出目录，不覆盖本索引结果。
"""


def size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def is_failed_json(path: Path) -> bool:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(obj, dict):
        return False
    state = str(obj.get("state", obj.get("status", ""))).lower()
    if state in {"failed", "fail", "error"} or obj.get("passed") is False:
        return True
    return False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def validate():
    if not BASE.is_dir() or ROOT.name != "Hexapod-MJLab":
        raise RuntimeError(f"Unexpected workspace root: {ROOT}")
    combined = set(MOVE) | set(DROP) | set(PROTECT)
    if len(combined) != len(MOVE) + len(DROP) + len(PROTECT):
        raise RuntimeError("A path is classified twice")
    # Must be first-time cleanup to avoid a partial double-apply.
    for name in combined:
        p = BASE / name
        if not p.is_dir() or p.is_symlink():
            raise RuntimeError(f"Expected original real directory absent/unsafe: {p}")
    if len(set(MOVE.values())) != len(MOVE):
        raise RuntimeError("Duplicate destination in rename plan")
    for new in (*MOVE.values(), *GPU_LINKS.keys()):
        path = BASE / new
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Destination already exists: {path}")
    for alias, target in GPU_LINKS.items():
        if target not in PROTECT or Path(alias).is_absolute() or ".." in Path(alias).parts:
            raise RuntimeError(f"Unsafe GPU navigation link: {alias} -> {target}")
    for old, new in MOVE.items():
        if old == new:
            raise RuntimeError(f"No-op rename: {old}")
        original = BASE / old
        if not any(original.iterdir()):
            raise RuntimeError(f"Unexpected empty source directory: {old}")
        for oldname, newname in FIGURES.items():
            if (original / oldname).is_file() and (original / newname).exists():
                raise RuntimeError(f"Figure destination collision: {old}/{newname}")
    for original in (*MOVE, *DROP):
        p = BASE / original
        if any(x.is_symlink() for x in p.rglob("*")):
            raise RuntimeError(f"Unexpected symlink in candidate directory: {p}")
    # Only known completed version should be removed, never an unknown newer
    # future run with the same prefix.
    for name in DROP:
        if name.startswith("confined_cpu-"):
            summary = BASE/name/"summary.json"
            data = json.loads(summary.read_text())
            if data.get("passed") is not True:
                raise RuntimeError(f"CPU run must PASS before replacement: {name}")
    for old,new in MOVE.items():
        if Path(old).name != old or Path(new).is_absolute() or ".." in Path(new).parts:
            raise RuntimeError("Unsupported unsafe move")
    if (BASE / "verification_organize_manifest.json").exists() or (
        BASE / "verification_organize_pending.json"
    ).exists():
        raise RuntimeError("Previous cleanup or interrupted cleanup exists; refusing double apply")
    for old in DROP:
        # Deleting referenced experiment evidence would silently break
        # source code or current authority docs, so reject it up front.
        needle = f"logs/verification/{old}"
        for rootdir in ("src", "tests", "scripts", "docs"):
            for path in (ROOT / rootdir).rglob("*"):
                if not path.is_file() or path.is_symlink() or path.suffix not in (".py", ".md"):
                    continue
                if path == ROOT / "scripts/tools/organize_verification_logs.py":
                    continue
                if needle in path.read_text(encoding="utf-8"):
                    raise RuntimeError(f"Refusing deletion of still-referenced result {old}: {path}")


def make_substitutions():
    """Only absolute project-relative evidence paths, not arbitrary filenames.

    Explicit figure-path replacements precede directory-prefix replacements.
    Bare labels in unrelated historical reports must remain untouched.
    """
    subs = {}
    for old, new in MOVE.items():
        for orig, newname in FIGURES.items():
            if (BASE / old / orig).is_file():
                subs[f"logs/verification/{old}/{orig}"] = (
                    f"logs/verification/{new}/{newname}"
                )
        subs[f"logs/verification/{old}"] = f"logs/verification/{new}"
    return subs


def source_rewrites(subs):
    files = []
    for filename in SOURCE_ALLOWLIST:
        path = ROOT / filename
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Reviewed source authority missing/unsafe: {path}")
        content = path.read_text(encoding="utf-8")
        newer = content
        for old, new in subs.items():
            newer = newer.replace(old, new)
        if filename.endswith("lidar_setup/配置与边界.md"):
            # This line references the MOVED v6 pose study. The v3 ground
            # figures elsewhere have the same old names but are NOT moved.
            old_line = "图 `pose_wall_fps_comparison.png` 与 `perturbation_group_changes.png`"
            new_line = "图 `sidewall_fps_by_robot_pose.png` 与 `range_gap_group_change_by_scan.png`"
            if old_line not in newer:
                raise RuntimeError("Expected v6 image caption not found; refusing blind rewrite")
            newer = newer.replace(old_line, new_line)
        if newer != content:
            files.append((path, content, newer))
    return files


def receipts_for_deletion():
    """Preserve failure summaries, underlying run recipes AND traceback logs.

    A former FAIL run can have the traceback in its root .log while the
    failure marker is nested in its report.json. If any structured report
    marks it failed, archive ALL small text receipts in that single run,
    never its heavyweight tensors/plots.
    """
    receipts = []
    for name in DROP:
        path = BASE / name
        has_failure = any(
            f.is_file() and f.suffix == ".json" and is_failed_json(f)
            for f in path.rglob("*")
        )
        if not has_failure:
            continue
        for f in path.rglob("*"):
            if f.is_file() and f.suffix in RECEIPT_SUFFIXES:
                receipts.append((name, f))
    return sorted(set(receipts), key=lambda x: (x[0], str(x[1])))


def protected_artifact_hashes():
    """Small set of untouched authoritative artifacts; prove cleanup did not mutate them."""
    s1 = BASE / "stageA_S1_187_formal_gpu_20261019_v1"
    checkpoints = list(s1.rglob("final_checkpoint_full_000001.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one S1-187 authority checkpoint, got {len(checkpoints)}")
    paths = (
        checkpoints[0],
        BASE/"feature-cache-20261013/final_report.json",
        BASE/"stageA_B8_lifecycle_gpu_20261019_v1/summary.json",
    )
    if not all(p.is_file() and not p.is_symlink() for p in paths):
        raise RuntimeError("Expected protected authority file missing/unsafe")
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def verify_completed():
    """Read-only verification of the complete renaming/cleanup outcome."""
    manifest_path = BASE/"verification_organize_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("No completed cleanup manifest found")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("schema")!="hexapod_mjlab.verification_organization/v1":
        raise RuntimeError("Unexpected cleanup manifest schema")
    if report.get("moved_directory_map")!=MOVE:
        raise RuntimeError("Manifest move mapping differs from current version")
    for old, new in MOVE.items():
        if (BASE/old).exists() or (BASE/old).is_symlink():
            raise RuntimeError(f"Stale un-moved source directory: {old}")
        if not (BASE/new).is_dir() or (BASE/new).is_symlink():
            raise RuntimeError(f"Missing destination directory: {new}")
    for name in DROP:
        if (BASE/name).exists() or (BASE/name).is_symlink():
            raise RuntimeError(f"Obsolete directory still present: {name}")
    for alias, original in GPU_LINKS.items():
        symlink=BASE/alias
        if not symlink.is_symlink() or symlink.resolve()!=(BASE/original).resolve():
            raise RuntimeError(f"GPU verification link is broken or wrong: {alias}")
    for name in PROTECT:
        if not (BASE/name).is_dir() or (BASE/name).is_symlink():
            raise RuntimeError(f"Protected evidence missing: {name}")
    if report.get("protected_file_sha256") != protected_artifact_hashes():
        raise RuntimeError("Protected checkpoint/authority SHA256 has changed")
    for receipt in report["old_failure_receipts_archived"]:
        if not (ROOT/receipt).is_file():
            raise RuntimeError(f"Archived failure receipt missing: {receipt}")
    for item in report["images_renamed"]:
        if not (ROOT/item["to"]).is_file():
            raise RuntimeError(f"Renamed diagnostic image missing: {item['to']}")
    if not (BASE/"README.md").is_file():
        raise RuntimeError("Verification navigation README missing")
    if (BASE/"verification_organize_pending.json").exists():
        raise RuntimeError("Cleanup pending marker was not cleared")
    print(f"[VERIFIED] {len(MOVE)} directories renamed; {len(DROP)} superseded directories gone")
    print(f"[VERIFIED] {len(GPU_LINKS)} GPU links; {len(report['old_failure_receipts_archived'])} retained text receipts")
    print("[VERIFIED] authority data directories preserved, manifest and index present")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan",action="store_true",help="read-only dry run (recommended first)")
    g.add_argument("--apply",action="store_true",help="explicitly perform planned rename/delete")
    g.add_argument("--verify",action="store_true",help="read-only verification after --apply")
    args=p.parse_args(argv)
    if args.verify:
        return verify_completed()
    validate()
    subs=make_substitutions()
    rewrites=source_rewrites(subs)
    receipts=receipts_for_deletion()
    moves_bytes=sum(size(BASE/old) for old in MOVE)
    deletes_bytes=sum(size(BASE/old) for old in DROP)
    print("=== Verification cleanup ===")
    print(f"Stage-A+CPU moves: {len(MOVE)} dirs; {moves_bytes/2**20:.1f} MiB")
    print(f"Superseded directories: {len(DROP)}; {deletes_bytes/2**20:.1f} MiB")
    print(f"FAIL receipts to archive: {len(receipts)}")
    print(f"Source/doc files with links updated: {len(rewrites)}")
    print(f"Navigation symlinks for protected GPU evidence: {len(GPU_LINKS)}")
    print("Checkpoint/recipe original absolute paths will remain unchanged.")
    for old,new in MOVE.items():
        print(f"MOVE {old} -> {new}")
    for old in DROP:
        print(f"DROP {old}")
    print("PROTECTED (will not touch):",", ".join(PROTECT))
    if args.plan:
        print("READ-ONLY: no changes performed. Apply only if classification matches your intention.")
        return 0

    protected_hashes=protected_artifact_hashes()
    # Persist the intended mapping before any move; if interrupted the
    # pending record blocks automatic retries until manually reconciled.
    pending = BASE / "verification_organize_pending.json"
    pending.write_text(json.dumps({
        "schema":"hexapod_mjlab.verification_cleanup_pending/v1",
        "started_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "moved_directory_map":MOVE,
        "deleted_directory_names":DROP,
        "gpu_navigation_links":GPU_LINKS,
        "protected_file_sha256":protected_hashes,
        "planned_source_edits":[str(p.relative_to(ROOT)) for p,_,_ in rewrites],
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    # Apply stage 1: rescue unique negative witnesses, then rename evidence.
    archive=BASE/"retained_failures"
    saved=[]
    for name,origin in receipts:
        dest=archive/name/origin.relative_to(BASE/name)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(origin,dest)
        if sha(origin)!=sha(dest):
            raise RuntimeError(f"Receipt copying checksum mismatch: {origin}")
        saved.append(str(dest.relative_to(ROOT)))
    for old,new in MOVE.items():
        dest=BASE/new
        dest.parent.mkdir(parents=True,exist_ok=True)
        (BASE/old).rename(dest)

    # Apply stage 2: make human-facing plot names match the index. Replace
    # only explicitly known figure names, never arbitrary file extensions.
    image_changes=[]
    for name in MOVE.values():
        destination=BASE/name
        for pth in tuple(destination.rglob("*")):
            if not pth.is_file() or pth.name not in FIGURES:
                continue
            new=pth.with_name(FIGURES[pth.name])
            if new.exists():
                raise RuntimeError(f"Figure name collision: {new}")
            pth.rename(new)
            image_changes.append({
                "from":str(pth.relative_to(ROOT)),
                "to":str(new.relative_to(ROOT))
            })

    # Create user-facing navigation links to the original GPU checkpoint
    # directories. NEVER change the actual GPU run paths.
    linked=[]
    for virtual,original in GPU_LINKS.items():
        alias=BASE/virtual
        target=BASE/original
        alias.parent.mkdir(parents=True,exist_ok=True)
        alias.symlink_to(os.path.relpath(target,start=alias.parent),target_is_directory=True)
        if alias.resolve()!=target.resolve():
            raise RuntimeError(f"GPU link resolution failed: {alias}")
        linked.append({"link":virtual,"target":original})

    # Only reviewed authority docs/analysis defaults change. The frozen
    # JSON/.pt remains byte-identical; report SHA refers to its earlier source.
    changed=[]
    for path,original,replacement in rewrites:
        if path.read_text(encoding="utf-8") != original:
            raise RuntimeError(f"Source edited during cleanup; aborted: {path}")
        path.write_text(replacement,encoding="utf-8")
        changed.append(str(path.relative_to(ROOT)))

    # Finally remove superseded directories. DO NOT touch gpu checkpoint
    # runs/feature-cache or original B8 ground-support v3 source.
    for old in DROP:
        shutil.rmtree(BASE/old)

    manifest = {
        "schema":"hexapod_mjlab.verification_organization/v1",
        "applied_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope":"logs/verification explicit stage-A and repeated CPU runs",
        "moved_directory_map":MOVE,
        "deleted_old_directories":DROP,
        "deleted_bytes_before_cleanup":deletes_bytes,
        "old_failure_receipts_archived":saved,
        "images_renamed":image_changes,
        "gpu_navigation_symlinks":linked,
        "protected_file_sha256":protected_hashes,
        "documents_and_scripts_with_updated_paths":changed,
        "protected_untouched_directories":PROTECT,
        "caveat":"Frozen JSON remains byte-identical; its historical original paths may differ from the physical relocated directory. Map old->new above. Do not use a renamed full checkpoint path; checkpoints were NOT moved.",
    }
    (BASE/"verification_organize_manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    (BASE/"README.md").write_text(INDEX,encoding="utf-8")
    pending.unlink()
    verify_completed()
    print(f"DONE. Deleted ~{deletes_bytes/2**20:.1f} MiB; "
          f"manifest: {BASE/'verification_organize_manifest.json'}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
