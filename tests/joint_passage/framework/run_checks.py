#!/usr/bin/env python
"""F1–F4 与语义修正轮统一 CPU 检查入口：依次运行五个 test_*.py 并以非零退出传播失败。

* 子进程统一使用当前解释器（sys.executable）与 CPU（子进程环境强制
  CUDA_VISIBLE_DEVICES=""，五个检查本身也只写 CPU 路径）。
* 只执行本目录下固定的五个文件名（不做任意路径脚本执行）；文件缺失即报错。
* 判定只依据子进程**实际退出码**：任一非零 → 本脚本退出 1；不解析、不依赖
  任何 PASS 文本字段；子进程启动异常（如 OSError）直接抛出，不吞异常。
* --report PATH（可选）：把本次实际退出码、用例数（尽力解析自 unittest 的
  "Ran N tests" 行，仅作记录、不作判定）与完整 stdout/stderr 日志写入
  Markdown 汇总。目标文件已存在时**拒绝覆盖旧证据**并以退出码 2 失败；
  父目录须已存在。写入用排他创建 open(..., "x")：预检查通过之后、实际写入
  之前被其他进程创建同路径文件（TOCTOU）同样拒绝覆盖（FileExistsError 明确
  处理、保留并发文件、返回 2），不采用 write_text 类覆盖写。默认不写任何
  文件、不改任何源码或旧 runner。
* 仅标准库；无新增依赖。用法（hp_mjlab 环境）：

    conda activate hp_mjlab
    python tests/joint_passage/framework/run_checks.py
    python tests/joint_passage/framework/run_checks.py --report /path/to/report.md
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
TEST_FILES = (
    "test_data.py",         # F1 数据契约与扫描窗口
    "test_network.py",      # F2 网络与形态输出
    "test_execution.py",    # F3 执行与接触
    "test_integration.py",  # F4 集成链路
    "test_semantics.py",    # 语义修正轮：映射/PD/远端元数据的工程数值正确性
)

_RAN_PATTERN = re.compile(r"^Ran (\d+) tests?", re.MULTILINE)


def _parse_case_count(*texts: str) -> int | None:
    """尽力解析 unittest 的 “Ran N tests” 行（仅报告记录用，不作通过判定）。"""
    for text in texts:
        match = _RAN_PATTERN.search(text)
        if match:
            return int(match.group(1))
    return None


def _build_report(started: str, outcomes: list[tuple[str, int, int | None, float, str, str]]) -> str:
    lines = [
        "# joint_passage 框架检查报告（F1–F4+语义修正轮）",
        "",
        f"- 开始时间: {started}",
        f"- 解释器: {sys.executable}",
        f"- 命令: {' '.join([sys.executable] + sys.argv)}",
        f"- 工作目录: {TEST_DIR}",
        "- CPU 强制: 子进程环境 CUDA_VISIBLE_DEVICES=\"\"",
        "",
        "## 汇总",
        "",
        "| 文件 | 退出码 | 用例数(解析) | 耗时(s) |",
        "|---|---|---|---|",
    ]
    for name, code, cases, seconds, _out, _err in outcomes:
        lines.append(f"| {name} | {code} | {cases if cases is not None else '未知'} | {seconds:.2f} |")
    failed = [name for name, code, *_ in outcomes if code != 0]
    lines += ["", "## 结论", ""]
    if failed:
        lines.append(f"失败（以实际退出码为准）: {', '.join(failed)}")
    else:
        lines.append("全部通过（以实际退出码为准）")
    lines += ["", "## 明细", ""]
    for name, code, cases, seconds, out, err in outcomes:
        lines += [
            f"### {name}",
            "",
            f"- 退出码: {code}",
            f"- 用例数(解析自 unittest 输出): {cases if cases is not None else '未知'}",
            f"- 耗时: {seconds:.2f}s",
            "",
            "stdout:",
            "",
            "```",
            out.rstrip("\n"),
            "```",
            "",
            "stderr:",
            "",
            "```",
            err.rstrip("\n"),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="依次运行 joint_passage 框架 F1–F4 与语义修正轮的五个 CPU 检查并传播失败。"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="可选：保存 Markdown 汇总（实际退出码/用例数/日志）；目标已存在则拒绝覆盖。",
    )
    args = parser.parse_args(argv)
    if args.report is not None:
        if args.report.exists():
            print(f"拒绝覆盖已有报告（保留旧证据）: {args.report}", file=sys.stderr)
            return 2
        if not args.report.parent.is_dir():
            print(f"报告父目录不存在（不默认创建）: {args.report.parent}", file=sys.stderr)
            return 2

    started = datetime.now().isoformat(timespec="seconds")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""   # 仅影响子进程：强制 CPU 路径
    outcomes: list[tuple[str, int, int | None, float, str, str]] = []
    for name in TEST_FILES:
        path = TEST_DIR / name
        if not path.is_file():
            print(f"缺少检查文件: {path}", file=sys.stderr)
            return 2
        print(f"===== {name} =====", flush=True)
        begin = time.monotonic()
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(TEST_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        seconds = time.monotonic() - begin
        out = completed.stdout or ""
        err = completed.stderr or ""
        if out:
            print(out, end="", flush=True)
        if err:
            print(err, end="", file=sys.stderr, flush=True)
        cases = _parse_case_count(err, out)
        outcomes.append((name, completed.returncode, cases, seconds, out, err))
        print(f"----- {name} 退出码 {completed.returncode}，"
              f"用例数(解析) {cases if cases is not None else '未知'} -----", flush=True)

    if args.report is not None:
        # 排他创建：即使上方预检查通过，写入前被并发进程创建同路径文件也拒绝
        # 覆盖（保留并发证据，FileExistsError 明确处理并返回 2，不静默改写）。
        try:
            with open(args.report, "x", encoding="utf-8") as handle:
                handle.write(_build_report(started, outcomes))
        except FileExistsError:
            print(f"报告路径在运行期间被其他进程创建，拒绝覆盖并发证据（保留该文件）: "
                  f"{args.report}", file=sys.stderr)
            return 2
        print(f"报告已写入: {args.report}")

    failed = [(name, code) for name, code, *_ in outcomes if code != 0]
    if failed:
        print("失败: " + ", ".join(f"{name}(退出码 {code})" for name, code in failed),
              file=sys.stderr)
        return 1
    print("全部通过: " + ", ".join(name for name, *_ in outcomes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
