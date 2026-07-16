#!/usr/bin/env python3
"""Generate a concise summary report on rollout tracking data.

This script queries the database and generates a summary-only report with:
- Overall statistics
- Data integrity checks
- Recommendations

No detailed per-rollout or per-trajectory information.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import asyncpg


async def generate_summary_report(
    host: str = None,
    port: int = None,
    database: str = None,
    user: str = None,
    password: str = None
):
    """Generate concise rollout tracking summary report.

    Args:
        host: Database host (defaults to env var DB_HOST or localhost)
        port: Database port (defaults to env var DB_PORT or 5433)
        database: Database name (defaults to env var DB_NAME or gymkhana)
        user: Database user (defaults to env var DB_USER or db_user)
        password: Database password (defaults to env var DB_PASSWORD)
    """

    # Use provided values or fall back to env vars
    host = host or os.getenv("DB_HOST", "localhost")
    port = port or int(os.getenv("DB_PORT", "5433"))
    database = database or os.getenv("DB_NAME", "gymkhana")
    user = user or os.getenv("DB_USER", "db_user")
    password = password or os.getenv("DB_PASSWORD", "db_pwd@123")

    print("=" * 80)
    print("ROLLOUT TRACKING SUMMARY")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {database} on {host}:{port}")
    print("=" * 80)
    print()

    try:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

        # =====================================================================
        # SUMMARY STATISTICS
        # =====================================================================
        print("📈 SUMMARY STATISTICS")
        print("-" * 80)

        # Get rollout-tracked data
        summary = await conn.fetchrow("""
            SELECT
                COUNT(DISTINCT rg.id) as total_groups,
                COUNT(DISTINCT r.id) as total_rollouts,
                COUNT(DISTINCT t.id) as total_trajectories,
                AVG(rg.reward_mean) as avg_group_mean_reward,
                AVG(rg.best_reward) as avg_best_reward,
                SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) as total_completed,
                SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) as total_failed,
                SUM(CASE WHEN r.status = 'error' THEN 1 ELSE 0 END) as total_error,
                SUM(CASE WHEN r.status = 'timeout' THEN 1 ELSE 0 END) as total_timeout,
                SUM(r.num_format_violations) as total_format_violations,
                AVG(r.num_turns) as avg_turns,
                AVG(r.num_code_blocks) as avg_code_blocks,
                SUM(CASE WHEN t.answer_correct = true THEN 1 ELSE 0 END) as correct_answers,
                SUM(CASE WHEN t.answer_correct = false THEN 1 ELSE 0 END) as incorrect_answers,
                SUM(CASE WHEN t.answer_correct IS NULL THEN 1 ELSE 0 END) as unknown_answers
            FROM rollout_groups rg
            LEFT JOIN rollouts r ON r.rollout_group_id = rg.id
            LEFT JOIN trajectories t ON t.rollout_id = r.id
        """)

        # Get standalone trajectories (not part of rollout groups)
        standalone = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_standalone,
                AVG(total_reward) as avg_reward,
                SUM(CASE WHEN answer_correct = true THEN 1 ELSE 0 END) as correct_answers,
                SUM(CASE WHEN answer_correct = false THEN 1 ELSE 0 END) as incorrect_answers,
                SUM(CASE WHEN answer_correct IS NULL THEN 1 ELSE 0 END) as unknown_answers,
                AVG(num_turns) as avg_turns,
                AVG(num_code_blocks) as avg_code_blocks
            FROM trajectories
            WHERE rollout_group_id IS NULL
        """)

        # Combine metrics
        total_trajectories = (summary['total_trajectories'] or 0) + (standalone['total_standalone'] or 0)
        total_correct = (summary['correct_answers'] or 0) + (standalone['correct_answers'] or 0)
        total_incorrect = (summary['incorrect_answers'] or 0) + (standalone['incorrect_answers'] or 0)
        total_unknown = (summary['unknown_answers'] or 0) + (standalone['unknown_answers'] or 0)

        print(f"\nOverall Metrics:")
        print(f"  Total Groups:              {summary['total_groups']}")
        print(f"  Total Rollouts:            {summary['total_rollouts']}")
        print(f"  Total Trajectories:        {total_trajectories}")
        if standalone['total_standalone']:
            print(f"    - Rollout-tracked:       {summary['total_trajectories'] or 0}")
            print(f"    - Standalone:            {standalone['total_standalone']}")

        print(f"\nReward Metrics:")
        avg_mean = summary['avg_group_mean_reward'] or 0.0
        avg_best = summary['avg_best_reward'] or 0.0
        print(f"  Avg Group Mean Reward:     {avg_mean:.4f}")
        print(f"  Avg Best Reward:           {avg_best:.4f}")

        print(f"\nStatus Distribution:")
        print(f"  Completed:                 {summary['total_completed']}")
        print(f"  Failed:                    {summary['total_failed']}")
        print(f"  Error:                     {summary['total_error']}")
        print(f"  Timeout:                   {summary['total_timeout']}")

        print(f"\nAnswer Correctness:")
        correct = summary['correct_answers'] or 0
        incorrect = summary['incorrect_answers'] or 0
        unknown = summary['unknown_answers'] or 0
        print(f"  Correct:                   {correct}")
        print(f"  Incorrect:                 {incorrect}")
        print(f"  Unknown:                   {unknown}")
        total_evaluated = correct + incorrect
        if total_evaluated > 0:
            accuracy = correct / total_evaluated * 100
            print(f"  Accuracy:                  {accuracy:.1f}% (of {total_evaluated} evaluated)")

        print(f"\nQuality Metrics:")
        total_violations = summary['total_format_violations'] or 0
        print(f"  Total Format Violations:   {total_violations}")
        if summary['total_rollouts'] and summary['total_rollouts'] > 0:
            violation_rate = total_violations / summary['total_rollouts'] * 100
            print(f"  Format Violation Rate:     {violation_rate:.1f}%")
        avg_turns = summary['avg_turns'] or 0.0
        avg_blocks = summary['avg_code_blocks'] or 0.0
        print(f"  Avg Turns per Rollout:     {avg_turns:.1f}")
        print(f"  Avg Code Blocks:           {avg_blocks:.1f}")

        # =====================================================================
        # DATA INTEGRITY CHECKS
        # =====================================================================
        print("\n" + "=" * 80)
        print("🔍 DATA INTEGRITY CHECKS")
        print("-" * 80)

        integrity_issues = []

        # Check 1: Rollout count matches group.num_rollouts
        mismatches = await conn.fetch("""
            SELECT COUNT(*) as count
            FROM rollout_groups rg
            LEFT JOIN rollouts r ON r.rollout_group_id = rg.id
            GROUP BY rg.id, rg.num_rollouts
            HAVING COUNT(r.id) != rg.num_rollouts
        """)
        if mismatches:
            integrity_issues.append(f"Rollout count mismatches: {len(mismatches)} groups")

        # Check 2: Trajectory-Rollout bidirectional linking
        broken_links = await conn.fetch("""
            SELECT COUNT(*) as count
            FROM trajectories t
            LEFT JOIN rollouts r ON r.id = t.rollout_id
            WHERE t.rollout_id IS NOT NULL
              AND (r.id IS NULL OR r.trajectory_id != t.id)
        """)
        if broken_links and broken_links[0]['count'] > 0:
            integrity_issues.append(f"Broken trajectory-rollout links: {broken_links[0]['count']}")

        # Check 3: Status counts match actual rollout statuses
        status_mismatches = await conn.fetch("""
            SELECT COUNT(*) as count
            FROM rollout_groups rg
            LEFT JOIN rollouts r ON r.rollout_group_id = rg.id
            GROUP BY rg.id, rg.num_completed, rg.num_failed, rg.num_error, rg.num_timeout
            HAVING
                rg.num_completed != SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) OR
                rg.num_failed != SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) OR
                rg.num_error != SUM(CASE WHEN r.status = 'error' THEN 1 ELSE 0 END) OR
                rg.num_timeout != SUM(CASE WHEN r.status = 'timeout' THEN 1 ELSE 0 END)
        """)
        if status_mismatches:
            integrity_issues.append(f"Status count mismatches: {len(status_mismatches)} groups")

        # Check 4: Best rollout ID points to highest reward
        best_rollout_errors = await conn.fetch("""
            WITH best_actual AS (
                SELECT
                    rollout_group_id,
                    id as best_rollout_id,
                    total_reward,
                    ROW_NUMBER() OVER (PARTITION BY rollout_group_id ORDER BY total_reward DESC) as rn
                FROM rollouts
            )
            SELECT COUNT(*) as count
            FROM rollout_groups rg
            LEFT JOIN best_actual ba ON ba.rollout_group_id = rg.id AND ba.rn = 1
            WHERE rg.best_rollout_id != ba.best_rollout_id
               OR ABS(rg.best_reward - ba.total_reward) > 0.0001
        """)
        if best_rollout_errors and best_rollout_errors[0]['count'] > 0:
            integrity_issues.append(f"Best rollout ID errors: {best_rollout_errors[0]['count']} groups")

        # Check 5: Reward statistics accuracy
        stats_errors = await conn.fetch("""
            WITH actual_stats AS (
                SELECT
                    rollout_group_id,
                    AVG(total_reward) as actual_mean,
                    STDDEV(total_reward) as actual_std,
                    MIN(total_reward) as actual_min,
                    MAX(total_reward) as actual_max
                FROM rollouts
                GROUP BY rollout_group_id
            )
            SELECT COUNT(*) as count
            FROM rollout_groups rg
            LEFT JOIN actual_stats ast ON ast.rollout_group_id = rg.id
            WHERE
                ABS(rg.reward_mean - ast.actual_mean) > 0.001 OR
                ABS(COALESCE(rg.reward_std, 0) - COALESCE(ast.actual_std, 0)) > 0.001 OR
                ABS(rg.reward_min - ast.actual_min) > 0.001 OR
                ABS(rg.best_reward - ast.actual_max) > 0.001
        """)
        if stats_errors and stats_errors[0]['count'] > 0:
            integrity_issues.append(f"Reward statistics errors: {stats_errors[0]['count']} groups")

        if integrity_issues:
            print("\n❌ Issues detected:")
            for issue in integrity_issues:
                print(f"  - {issue}")
            print("\n  Run 'python scripts/generate_rollout_report.py' for detailed analysis")
        else:
            print("\n✅ All integrity checks passed")

        # =====================================================================
        # RECOMMENDATIONS
        # =====================================================================
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("-" * 80)

        recommendations = []

        # Check for high format violation rate
        if summary['total_rollouts'] > 0 and summary['total_format_violations'] > 0:
            violation_rate = summary['total_format_violations'] / summary['total_rollouts'] * 100
            if violation_rate > 10:
                recommendations.append(
                    f"⚠️  High format violation rate ({violation_rate:.1f}%). "
                    "Consider adjusting system prompts or model parameters."
                )

        # Check for high failure rate
        if summary['total_rollouts'] > 0 and summary['total_failed'] > 0:
            failure_rate = summary['total_failed'] / summary['total_rollouts'] * 100
            if failure_rate > 20:
                recommendations.append(
                    f"⚠️  High failure rate ({failure_rate:.1f}%). "
                    "Review termination policies - they may be too strict."
                )

        # Check for low accuracy
        correct = summary['correct_answers'] or 0
        incorrect = summary['incorrect_answers'] or 0
        total_evaluated = correct + incorrect
        if total_evaluated > 0:
            accuracy = correct / total_evaluated * 100
            if accuracy < 50:
                recommendations.append(
                    f"⚠️  Low accuracy ({accuracy:.1f}%). "
                    "Consider adjusting prompts, model, or task difficulty."
                )

        # Check for data integrity issues
        if integrity_issues:
            recommendations.append(
                "❌ Data integrity issues detected. "
                "Run 'python scripts/generate_rollout_report.py' for detailed analysis."
            )

        if not recommendations:
            print("\n✅ No issues detected. System is working as expected!")
        else:
            print()
            for rec in recommendations:
                print(f"  {rec}")

        print("\n" + "=" * 80)
        print("END OF SUMMARY")
        print("=" * 80)

        await conn.close()

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(generate_summary_report())
