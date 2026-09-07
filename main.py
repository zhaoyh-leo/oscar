#!/usr/bin/env python3
"""OSCAR - Open-Source Completeness Audit & Review.

Usage:
    python main.py <github_repo_url> [--paper <paper_url>]
    python main.py --help
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Windows 中文环境:stdout 被重定向/管道(日志、CI、IDE)时默认退回 gbk 编码,
# 打印 emoji(🔍✅ 等)会直接 UnicodeEncodeError 崩溃。统一强制 UTF-8 兜底。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
del _stream

from oscar.config import config
from oscar.graph.workflow import run_audit
from oscar.utils.progress_handler import RichProgressHandler


def main():
    parser = argparse.ArgumentParser(
        description="OSCAR - Open-Source Completeness Audit & Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py https://github.com/user/repo
    python main.py https://github.com/user/repo --paper https://arxiv.org/abs/1234.56789
    python main.py https://github.com/user/repo --output ./my_audit
        """,
    )
    parser.add_argument("repo_url", help="GitHub repository URL to audit")
    parser.add_argument("--paper", "-p", help="Paper URL or arXiv ID")
    parser.add_argument("--output", "-o", help="Output directory for reports", default=None)
    # --no-cleanup 需要区分「未给出」(交给 config.yaml/默认)与「显式关闭
    # 清理」:default=None,仅非 None 才覆盖 config
    parser.add_argument("--no-cleanup", action="store_true", default=None,
                        help="Keep cloned repository after audit (overrides config.yaml audit.cleanup_repo)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Configure
    if args.output:
        config.paths.output_dir = Path(args.output)
    if args.no_cleanup is not None:
        config.audit.cleanup_repo = not args.no_cleanup

    config.ensure_dirs()

    print(f"🔍 OSCAR - Auditing: {args.repo_url}")
    if args.paper:
        print(f"📄 Paper: {args.paper}")
    print("")

    try:
        # Progress handler with rich UI
        progress_handler = RichProgressHandler()

        # Run audit with progress callback (async graph → asyncio.run)
        final_state = asyncio.run(
            run_audit(args.repo_url, args.paper, progress_handler=progress_handler)
        )

        # Stop progress
        progress_handler.stop()

        # Print results
        report = final_state.report
        if report:
            print(f"\n✅ Audit Complete!")
            print(f"   Project: {report.project_name}")
            print(f"   Findings: {len(report.findings)}")
            print(f"   Uncertain: {len(report.uncertain_findings)}")
            print(f"")
            print(f"   Summary Statistics:")
            for status, count in sorted(report.summary_stats.items()):
                print(f"     {status}: {count}")
            print(f"")
            # Show project-specific output path
            out_dir = config.paths.output_dir / final_state.project.get("name", "unknown")
            print(f"   Reports saved to: {out_dir}")
            print(f"     - {out_dir / 'audit_report.md'}")
            print(f"     - {out_dir / 'audit_result.json'}")
            print(f"     - {out_dir / 'repository_manifest.json'}")
            # Also print code summary if available
            if final_state.repository_manifest and final_state.repository_manifest.code_summary:
                cs = final_state.repository_manifest.code_summary
                if cs.project_summary:
                    print(f"\n   Project Summary: {cs.project_summary}")
        else:
            print("\n❌ Audit failed: No report generated.")
            if final_state.errors:
                print("Errors:")
                for err in final_state.errors:
                    print(f"  - {err}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Audit failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()