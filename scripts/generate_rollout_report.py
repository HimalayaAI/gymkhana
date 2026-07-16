#!/usr/bin/env python3
"""Generate a comprehensive report on rollout tracking data.

This script queries the database and generates a detailed report on:
- Rollout groups
- Individual rollouts
- Trajectories
- Statistics and metrics
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


async def generate_report(
    host: str = None,
    port: int = None,
    database: str = None,
    user: str = None,
    password: str = None
):
    """Generate comprehensive rollout tracking report.

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
    print("ROLLOUT TRACKING REPORT")
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
        # 1. ROLLOUT GROUPS OVERVIEW
        # =====================================================================
        print("📊 ROLLOUT GROUPS OVERVIEW")
        print("-" * 80)

        groups = await conn.fetch("""
            SELECT
                id,
                task_id,
                environment,
                num_rollouts,
                num_completed,
                num_failed,
                num_error,
                num_timeout,
                best_reward,
                reward_mean,
                reward_std,
                reward_min,
                created_at,
                completed_at
            FROM rollout_groups
            ORDER BY created_at DESC
        """)

        if not groups:
            print("⚠️  No rollout groups found in database")
        else:
            print(f"Total rollout groups: {len(groups)}")
            print()

            for i, group in enumerate(groups, 1):
                print(f"Group {i}: {group['id']}")
                print(f"  Task ID:        {group['task_id']}")
                print(f"  Environment:    {group['environment']}")
                print(f"  Num Rollouts:   {group['num_rollouts']}")
                print(f"  Status Counts:")
                print(f"    - Completed:  {group['num_completed']}")
                print(f"    - Failed:     {group['num_failed']}")
                print(f"    - Error:      {group['num_error']}")
                print(f"    - Timeout:    {group['num_timeout']}")
                print(f"  Reward Stats:")
                print(f"    - Best:       {group['best_reward']:.4f}")
                print(f"    - Mean:       {group['reward_mean']:.4f}")
                print(f"    - Std Dev:    {group['reward_std']:.4f}")
                print(f"    - Min:        {group['reward_min']:.4f}")
                print(f"  Created:        {group['created_at']}")
                print(f"  Completed:      {group['completed_at']}")
                print()

        # =====================================================================
        # 2. ROLLOUTS DETAIL
        # =====================================================================
        print("=" * 80)
        print("🎯 ROLLOUTS DETAIL")
        print("-" * 80)

        rollouts = await conn.fetch("""
            SELECT
                r.id,
                r.rollout_group_id,
                r.rollout_index,
                r.status,
                r.termination_reason,
                r.num_turns,
                r.num_code_blocks,
                r.num_errors,
                r.consecutive_errors,
                r.num_format_violations,
                r.consecutive_format_violations,
                r.last_format_violation_type,
                r.total_reward,
                r.duration_ms,
                r.trajectory_id,
                rg.task_id
            FROM rollouts r
            JOIN rollout_groups rg ON rg.id = r.rollout_group_id
            ORDER BY r.rollout_group_id, r.rollout_index
        """)

        if not rollouts:
            print("⚠️  No rollouts found in database")
        else:
            print(f"Total rollouts: {len(rollouts)}")
            print()

            current_group = None
            for rollout in rollouts:
                if rollout['rollout_group_id'] != current_group:
                    current_group = rollout['rollout_group_id']
                    print(f"\n📦 Group: {current_group} (Task: {rollout['task_id']})")
                    print("-" * 80)

                print(f"  Rollout #{rollout['rollout_index']}:")
                print(f"    ID:                    {rollout['id']}")
                print(f"    Status:                {rollout['status']}")
                if rollout['termination_reason']:
                    print(f"    Termination Reason:    {rollout['termination_reason']}")
                print(f"    Turns:                 {rollout['num_turns']}")
                print(f"    Code Blocks:           {rollout['num_code_blocks']}")
                print(f"    Errors:                {rollout['num_errors']} (consecutive: {rollout['consecutive_errors']})")
                print(f"    Format Violations:     {rollout['num_format_violations']} (consecutive: {rollout['consecutive_format_violations']})")
                if rollout['last_format_violation_type']:
                    print(f"    Last Violation Type:   {rollout['last_format_violation_type']}")
                print(f"    Total Reward:          {rollout['total_reward']:.4f}")
                print(f"    Duration:              {rollout['duration_ms']:.0f}ms" if rollout['duration_ms'] else "    Duration:              N/A")
                print(f"    Trajectory ID:         {rollout['trajectory_id']}")
                print()

        # =====================================================================
        # 3. TRAJECTORIES LINKED TO ROLLOUTS
        # =====================================================================
        print("=" * 80)
        print("📝 TRAJECTORIES LINKED TO ROLLOUTS")
        print("-" * 80)

        trajectories = await conn.fetch("""
            SELECT
                t.id,
                t.rollout_id,
                t.rollout_group_id,
                t.rollout_index,
                t.task_id,
                t.success,
                t.answer_correct,
                t.num_code_blocks,
                t.num_errors,
                t.total_reward,
                t.final_answer
            FROM trajectories t
            WHERE t.rollout_group_id IS NOT NULL
            ORDER BY t.rollout_group_id, t.rollout_index
        """)

        if not trajectories:
            print("⚠️  No trajectories linked to rollouts found")
        else:
            print(f"Total trajectories with rollout tracking: {len(trajectories)}")
            print()

            current_group = None
            for traj in trajectories:
                if traj['rollout_group_id'] != current_group:
                    current_group = traj['rollout_group_id']
                    print(f"\n📦 Group: {current_group} (Task: {traj['task_id']})")
                    print("-" * 80)

                print(f"  Trajectory #{traj['rollout_index']}:")
                print(f"    ID:              {traj['id']}")
                print(f"    Rollout ID:      {traj['rollout_id']}")
                print(f"    Success:         {traj['success']}")
                print(f"    Answer Correct:  {traj['answer_correct']}")
                print(f"    Code Blocks:     {traj['num_code_blocks']}")
                print(f"    Errors:          {traj['num_errors']}")
                print(f"    Total Reward:    {traj['total_reward']:.4f}")
                print(f"    Final Answer:    {traj['final_answer'][:100]}..." if traj['final_answer'] and len(traj['final_answer']) > 100 else f"    Final Answer:    {traj['final_answer']}")
                print()

        # =====================================================================
        # 4. DATA INTEGRITY CHECKS
        # =====================================================================
        print("=" * 80)
        print("🔍 DATA INTEGRITY CHECKS")
        print("-" * 80)

        # Check 1: Rollout count matches group.num_rollouts
        print("\n✓ Check 1: Rollout count matches group.num_rollouts")
        mismatches = await conn.fetch("""
            SELECT
                rg.id as group_id,
                rg.num_rollouts as expected,
                COUNT(r.id) as actual
            FROM rollout_groups rg
            LEFT JOIN rollouts r ON r.rollout_group_id = rg.id
            GROUP BY rg.id, rg.num_rollouts
            HAVING COUNT(r.id) != rg.num_rollouts
        """)

        if mismatches:
            print(f"  ❌ Found {len(mismatches)} mismatches:")
            for m in mismatches:
                print(f"     Group {m['group_id']}: expected {m['expected']}, got {m['actual']}")
        else:
            print("  ✅ All groups have correct rollout count")

        # Check 2: Trajectory-Rollout bidirectional linking
        print("\n✓ Check 2: Trajectory-Rollout bidirectional linking")
        broken_links = await conn.fetch("""
            SELECT
                t.id as trajectory_id,
                t.rollout_id as traj_rollout_id,
                r.id as rollout_id,
                r.trajectory_id as rollout_traj_id
            FROM trajectories t
            LEFT JOIN rollouts r ON r.id = t.rollout_id
            WHERE t.rollout_id IS NOT NULL
              AND (r.id IS NULL OR r.trajectory_id != t.id)
        """)

        if broken_links:
            print(f"  ❌ Found {len(broken_links)} broken links:")
            for link in broken_links:
                print(f"     Trajectory {link['trajectory_id']} → Rollout {link['traj_rollout_id']}")
                print(f"     Rollout {link['rollout_id']} → Trajectory {link['rollout_traj_id']}")
        else:
            print("  ✅ All trajectory-rollout links are bidirectional")

        # Check 3: Status counts match actual rollout statuses
        print("\n✓ Check 3: Status counts match actual rollout statuses")
        status_mismatches = await conn.fetch("""
            SELECT
                rg.id as group_id,
                rg.num_completed,
                rg.num_failed,
                rg.num_error,
                rg.num_timeout,
                SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) as actual_completed,
                SUM(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END) as actual_failed,
                SUM(CASE WHEN r.status = 'error' THEN 1 ELSE 0 END) as actual_error,
                SUM(CASE WHEN r.status = 'timeout' THEN 1 ELSE 0 END) as actual_timeout
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
            print(f"  ❌ Found {len(status_mismatches)} status count mismatches:")
            for m in status_mismatches:
                print(f"     Group {m['group_id']}:")
                print(f"       Completed: {m['num_completed']} (expected) vs {m['actual_completed']} (actual)")
                print(f"       Failed:    {m['num_failed']} (expected) vs {m['actual_failed']} (actual)")
                print(f"       Error:     {m['num_error']} (expected) vs {m['actual_error']} (actual)")
                print(f"       Timeout:   {m['num_timeout']} (expected) vs {m['actual_timeout']} (actual)")
        else:
            print("  ✅ All status counts match actual rollout statuses")

        # Check 4: Best rollout ID points to highest reward
        print("\n✓ Check 4: Best rollout ID points to highest reward")
        best_rollout_errors = await conn.fetch("""
            WITH best_actual AS (
                SELECT
                    rollout_group_id,
                    id as best_rollout_id,
                    total_reward,
                    ROW_NUMBER() OVER (PARTITION BY rollout_group_id ORDER BY total_reward DESC) as rn
                FROM rollouts
            )
            SELECT
                rg.id as group_id,
                rg.best_rollout_id as expected_best,
                rg.best_reward as expected_reward,
                ba.best_rollout_id as actual_best,
                ba.total_reward as actual_reward
            FROM rollout_groups rg
            LEFT JOIN best_actual ba ON ba.rollout_group_id = rg.id AND ba.rn = 1
            WHERE rg.best_rollout_id != ba.best_rollout_id
               OR ABS(rg.best_reward - ba.total_reward) > 0.0001
        """)

        if best_rollout_errors:
            print(f"  ❌ Found {len(best_rollout_errors)} best rollout mismatches:")
            for err in best_rollout_errors:
                print(f"     Group {err['group_id']}:")
                print(f"       Expected: {err['expected_best']} (reward: {err['expected_reward']:.4f})")
                print(f"       Actual:   {err['actual_best']} (reward: {err['actual_reward']:.4f})")
        else:
            print("  ✅ All best_rollout_id values point to highest reward")

        # Check 5: Reward statistics accuracy
        print("\n✓ Check 5: Reward statistics accuracy")
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
            SELECT
                rg.id as group_id,
                rg.reward_mean,
                rg.reward_std,
                rg.reward_min,
                rg.best_reward,
                ast.actual_mean,
                ast.actual_std,
                ast.actual_min,
                ast.actual_max
            FROM rollout_groups rg
            LEFT JOIN actual_stats ast ON ast.rollout_group_id = rg.id
            WHERE
                ABS(rg.reward_mean - ast.actual_mean) > 0.001 OR
                ABS(COALESCE(rg.reward_std, 0) - COALESCE(ast.actual_std, 0)) > 0.001 OR
                ABS(rg.reward_min - ast.actual_min) > 0.001 OR
                ABS(rg.best_reward - ast.actual_max) > 0.001
        """)

        if stats_errors:
            print(f"  ❌ Found {len(stats_errors)} reward statistics errors:")
            for err in stats_errors:
                print(f"     Group {err['group_id']}:")
                print(f"       Mean:     {err['reward_mean']:.4f} (stored) vs {err['actual_mean']:.4f} (actual)")
                print(f"       Std Dev:  {err['reward_std']:.4f} (stored) vs {err['actual_std']:.4f} (actual)")
                print(f"       Min:      {err['reward_min']:.4f} (stored) vs {err['actual_min']:.4f} (actual)")
                print(f"       Max:      {err['best_reward']:.4f} (stored) vs {err['actual_max']:.4f} (actual)")
        else:
            print("  ✅ All reward statistics are accurate")

        # =====================================================================
        # 5. SUMMARY STATISTICS
        # =====================================================================
        print("\n" + "=" * 80)
        print("📈 SUMMARY STATISTICS")
        print("-" * 80)

        summary = await conn.fetchrow("""
            SELECT
                COUNT(DISTINCT rg.id) as total_groups,
                COUNT(DISTINCT r.id) as total_rollouts,
                COUNT(DISTINCT t.id) as total_trajectories,
                AVG(rg.reward_mean) as avg_group_mean_reward,
                AVG(rg.best_reward) as avg_best_reward,
                SUM(rg.num_completed) as total_completed,
                SUM(rg.num_failed) as total_failed,
                SUM(rg.num_error) as total_error,
                SUM(rg.num_timeout) as total_timeout,
                SUM(r.num_format_violations) as total_format_violations,
                AVG(r.num_turns) as avg_turns,
                AVG(r.num_code_blocks) as avg_code_blocks
            FROM rollout_groups rg
            LEFT JOIN rollouts r ON r.rollout_group_id = rg.id
            LEFT JOIN trajectories t ON t.rollout_group_id = rg.id
        """)

        print(f"\nOverall Metrics:")
        print(f"  Total Groups:              {summary['total_groups']}")
        print(f"  Total Rollouts:            {summary['total_rollouts']}")
        print(f"  Total Trajectories:        {summary['total_trajectories']}")
        print(f"\nReward Metrics:")
        print(f"  Avg Group Mean Reward:     {summary['avg_group_mean_reward']:.4f}")
        print(f"  Avg Best Reward:           {summary['avg_best_reward']:.4f}")
        print(f"\nStatus Distribution:")
        print(f"  Completed:                 {summary['total_completed']}")
        print(f"  Failed:                    {summary['total_failed']}")
        print(f"  Error:                     {summary['total_error']}")
        print(f"  Timeout:                   {summary['total_timeout']}")
        print(f"\nQuality Metrics:")
        print(f"  Total Format Violations:   {summary['total_format_violations']}")
        print(f"  Avg Turns per Rollout:     {summary['avg_turns']:.1f}")
        print(f"  Avg Code Blocks:           {summary['avg_code_blocks']:.1f}")

        # =====================================================================
        # 6. RECOMMENDATIONS
        # =====================================================================
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATIONS")
        print("-" * 80)

        recommendations = []

        # Check for high format violation rate
        if summary['total_format_violations'] > 0:
            violation_rate = summary['total_format_violations'] / summary['total_rollouts'] * 100
            if violation_rate > 10:
                recommendations.append(
                    f"⚠️  High format violation rate ({violation_rate:.1f}%). "
                    "Consider adjusting system prompts or model parameters."
                )

        # Check for high failure rate
        if summary['total_failed'] > 0:
            failure_rate = summary['total_failed'] / summary['total_rollouts'] * 100
            if failure_rate > 20:
                recommendations.append(
                    f"⚠️  High failure rate ({failure_rate:.1f}%). "
                    "Review termination policies - they may be too strict."
                )

        # Check for data integrity issues
        if mismatches or broken_links or status_mismatches or best_rollout_errors or stats_errors:
            recommendations.append(
                "❌ Data integrity issues detected. Review the checks above and fix inconsistencies."
            )

        if not recommendations:
            print("\n✅ No issues detected. System is working as expected!")
        else:
            print()
            for rec in recommendations:
                print(f"  {rec}")

        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)

        await conn.close()

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(generate_report())
