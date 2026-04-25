#!/usr/bin/env python3
"""
ClickHouse Migration Automator — Load Balancer Compatible
==========================================================
Optimizes column types (Decimal256 → UInt/Int/Decimal32-128, LowCardinality,
codecs, etc.) via post-load profiling and partition-by-partition migration.

Works entirely through a single LB endpoint. No direct shard access needed.

Two table types (--table-type):

  fact  (sharded)
    Naming:   local = {name}_local  |  dist = {name}
    Data:     Sharded across 3 shards, replicated within each shard
    Migrate:  parallel_distributed_insert_select=2 through Distributed tables
              → each shard reads/writes its own data locally

  dim   (replicated full-copy)
    Naming:   local = {name}  |  dist = {name}_dist
    Data:     Full copy on all 6 nodes via ZK replication
    Migrate:  INSERT on ONE node (whichever LB picks)
              → ZK replicates to other 5 nodes
              ⚠ NEVER uses parallel_distributed_insert_select (would cause 6x duplication)

Usage:
    # Dry run a fact table
    python ch_lb_migrate.py \\
        --host lb.internal --database etl_db --cluster my_cluster \\
        --table app_fact_trades --table-type fact --dry-run

    # Dry run a dim table
    python ch_lb_migrate.py \\
        --host lb.internal --database etl_db --cluster my_cluster \\
        --table app_ref_currency --table-type dim --dry-run

    # Migrate multiple facts
    python ch_lb_migrate.py \\
        --host lb.internal --database etl_db --cluster my_cluster \\
        --tables app_fact_trades app_fact_positions --table-type fact

    # Auto-find all Decimal256 dim tables
    python ch_lb_migrate.py \\
        --host lb.internal --database etl_db --cluster my_cluster \\
        --all-decimal256 --table-type dim --dry-run

    # Save report
    python ch_lb_migrate.py \\
        --host lb.internal --database etl_db --cluster my_cluster \\
        --table app_fact_trades --table-type fact \\
        --output-report report.json --dry-run
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════════════════
try:
    import clickhouse_connect

    def make_client(host, port, user, password, database):
        return clickhouse_connect.get_client(
            host=host, port=port or 8123,
            username=user, password=password, database=database,
            # ⚠ Behind a LB, consecutive requests can land on different nodes.
            # clickhouse-connect uses persistent sessions by default, but the
            # session UUID is node-local — a second node sees it as "locked".
            # Setting session_id='' disables server-side sessions entirely.
            session_id='',
        )

    def q(client, sql, settings=None):
        r = client.query(sql, settings=settings or {})
        return [dict(zip(r.column_names, row)) for row in r.result_rows]

    def cmd(client, sql, settings=None):
        client.command(sql, settings=settings or {})

except ImportError:
    try:
        from clickhouse_driver import Client as _NC

        def make_client(host, port, user, password, database):
            return _NC(host=host, port=port or 9000,
                       user=user, password=password, database=database)

        def q(client, sql, settings=None):
            data, cols = client.execute(sql, with_column_types=True,
                                        settings=settings or {})
            return [dict(zip([c[0] for c in cols], row)) for row in data]

        def cmd(client, sql, settings=None):
            client.execute(sql, settings=settings or {})
    except ImportError:
        sys.exit("pip install clickhouse-connect OR clickhouse-driver")


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ColumnRec:
    name: str
    current_type: str
    recommended_type: str
    codec: str
    is_in_order_by: bool
    is_in_partition_by: bool
    changed: bool
    reason: str
    compressed_mb: float = 0
    uncompressed_mb: float = 0


@dataclass
class TableTopology:
    table_type: str                  # "fact" or "dim"
    base_name: str                   # user-supplied name (without _local / _dist)
    local_table: str                 # actual local table name
    distributed_table: str           # actual distributed table name
    database: str
    cluster: str
    engine_full: str                 # ENGINE = ReplicatedMergeTree(...)
    order_by: str
    partition_by: str
    settings_block: str
    sharding_key: str
    columns: list[ColumnRec] = field(default_factory=list)

    # Derived names for migration artifacts
    @property
    def local_v2(self) -> str:
        return f"{self.local_table}_v2"

    @property
    def local_old(self) -> str:
        return f"{self.local_table}_old"

    @property
    def dist_v2(self) -> str:
        """Temp Distributed table for v2 (facts only)."""
        if self.table_type == "fact":
            return f"{self.base_name}_v2"
        else:
            return f"{self.base_name}_v2_dist"


# ═══════════════════════════════════════════════════════════════════════════
# Naming conventions
# ═══════════════════════════════════════════════════════════════════════════
def resolve_names(table_name: str, table_type: str) -> tuple[str, str, str]:
    """
    Given user input and table type, return (base_name, local_table, dist_table).

    Fact:
      User passes: app_fact_trades  (the dist name, which is the "base")
      local = app_fact_trades_local
      dist  = app_fact_trades
      User can also pass: app_fact_trades_local → strips _local

    Dim:
      User passes: app_ref_currency  (the local name, which is the "base")
      local = app_ref_currency
      dist  = app_ref_currency_dist
      User can also pass: app_ref_currency_dist → strips _dist
    """
    if table_type == "fact":
        if table_name.endswith("_local"):
            base = table_name[:-6]
        else:
            base = table_name
        local = f"{base}_local"
        dist = base
    else:  # dim
        if table_name.endswith("_dist"):
            base = table_name[:-5]
        else:
            base = table_name
        local = base
        dist = f"{base}_dist"

    return base, local, dist


# ═══════════════════════════════════════════════════════════════════════════
# Topology discovery
# ═══════════════════════════════════════════════════════════════════════════
def discover_topology(client, db: str, table_name: str,
                      table_type: str, cluster: str) -> TableTopology:
    """Build topology from naming convention + SHOW CREATE."""

    base, local, dist = resolve_names(table_name, table_type)

    # Verify local table exists
    exists = q(client, f"""
        SELECT name, engine
        FROM system.tables
        WHERE database = '{db}' AND name = '{local}'
    """)
    if not exists:
        raise ValueError(f"Local table {db}.{local} not found")

    # Verify distributed table exists
    dist_exists = q(client, f"""
        SELECT name, engine, engine_full
        FROM system.tables
        WHERE database = '{db}' AND name = '{dist}'
    """)
    if not dist_exists:
        logger.warning(f"  Distributed table {db}.{dist} not found — will skip dist operations")
        dist = ""
        sharding_key = "rand()"
    else:
        eng_full = dist_exists[0]["engine_full"]
        m = re.search(r"Distributed\(([^)]+)\)", eng_full)
        if m:
            parts = [p.strip().strip("'\"") for p in m.group(1).split(",")]
            sharding_key = parts[3].strip() if len(parts) >= 4 else "rand()"
        else:
            sharding_key = "rand()"

    # Get CREATE TABLE of local table
    create_rows = q(client, f"SHOW CREATE TABLE `{db}`.`{local}`")
    create_sql = list(create_rows[0].values())[0]

    # Parse ORDER BY
    order_match = re.search(r'ORDER BY\s*\(([^)]+)\)', create_sql)
    if not order_match:
        order_match = re.search(r'ORDER BY\s+(.+?)(?:\n|SETTINGS|$)', create_sql)
    order_by = order_match.group(1).strip() if order_match else ""

    # Parse PARTITION BY
    part_match = re.search(r'PARTITION BY\s+(.+?)(?:\n|ORDER)', create_sql)
    partition_by = part_match.group(1).strip() if part_match else ""

    # Parse ENGINE
    eng_match = re.search(
        r"(ENGINE\s*=\s*(?:Replicated\w*MergeTree|MergeTree)\([^)]*\))",
        create_sql, re.DOTALL,
    )
    engine_full = eng_match.group(1).strip() if eng_match else "ENGINE = MergeTree()"

    # Parse SETTINGS
    settings_match = re.search(r'(SETTINGS\s+.+)$', create_sql, re.DOTALL)
    settings_block = (settings_match.group(1).strip() if settings_match
                      else "SETTINGS index_granularity = 8192")

    # Show cluster layout
    try:
        ci = q(client, f"""
            SELECT shard_num, replica_num, host_name, is_local
            FROM system.clusters WHERE cluster = '{cluster}'
            ORDER BY shard_num, replica_num
        """)
        logger.info(f"  Cluster '{cluster}':")
        for row in ci:
            tag = " ← connected" if row.get("is_local") else ""
            logger.info(f"    Shard {row['shard_num']} / Replica {row['replica_num']}: "
                        f"{row['host_name']}{tag}")
    except Exception:
        pass

    return TableTopology(
        table_type=table_type,
        base_name=base,
        local_table=local,
        distributed_table=dist,
        database=db,
        cluster=cluster,
        engine_full=engine_full,
        order_by=order_by,
        partition_by=partition_by,
        settings_block=settings_block,
        sharding_key=sharding_key,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Column profiling (post-load on CH data)
# ═══════════════════════════════════════════════════════════════════════════
def profile_columns(client, topo: TableTopology, sample_pct: float):
    """
    Profile columns for type optimization recommendations.

    For facts:  profiles through the Distributed table (cross-shard sample).
    For dims:   profiles through the local table (all nodes have full data).
    """
    db = topo.database

    if topo.table_type == "fact" and topo.distributed_table:
        profile_table = topo.distributed_table
    else:
        profile_table = topo.local_table

    # Parse ORDER BY / PARTITION BY columns
    order_by_cols = set()
    for part in topo.order_by.split(","):
        col = part.strip().strip("`").split("(")[-1].rstrip(")")
        order_by_cols.add(col)

    partition_by_cols = set()
    if topo.partition_by:
        for token in re.findall(r'`?(\w+)`?', topo.partition_by):
            partition_by_cols.add(token)

    # Column metadata
    col_meta = q(client, f"""
        SELECT name, type
        FROM system.columns
        WHERE database = '{db}' AND table = '{topo.local_table}'
        ORDER BY position
    """)

    # Column sizes (cluster-wide)
    sizes = {}
    try:
        size_rows = q(client, f"""
            SELECT
                column,
                sum(column_data_compressed_bytes)   AS comp,
                sum(column_data_uncompressed_bytes)  AS uncomp
            FROM clusterAllReplicas('{topo.cluster}', system.parts_columns)
            WHERE database = '{db}' AND table = '{topo.local_table}' AND active = 1
            GROUP BY column
        """)
        sizes = {r["column"]: (r["comp"], r["uncomp"]) for r in size_rows}
    except Exception:
        try:
            size_rows = q(client, f"""
                SELECT column,
                    sum(column_data_compressed_bytes) AS comp,
                    sum(column_data_uncompressed_bytes) AS uncomp
                FROM system.parts_columns
                WHERE database = '{db}' AND table = '{topo.local_table}' AND active = 1
                GROUP BY column
            """)
            sizes = {r["column"]: (r["comp"], r["uncomp"]) for r in size_rows}
        except Exception:
            pass

    columns = []
    for cm in col_meta:
        col_name, col_type = cm["name"], cm["type"]
        rec_type, codec, reason = _recommend_column(
            client, db, profile_table, col_name, col_type, sample_pct
        )
        comp, uncomp = sizes.get(col_name, (0, 0))

        columns.append(ColumnRec(
            name=col_name,
            current_type=col_type,
            recommended_type=rec_type,
            codec=codec,
            is_in_order_by=col_name in order_by_cols,
            is_in_partition_by=col_name in partition_by_cols,
            changed=rec_type != col_type,
            reason=reason,
            compressed_mb=round(comp / 1e6, 2),
            uncompressed_mb=round(uncomp / 1e6, 2),
        ))

    topo.columns = columns


def _sample_clause(sample_pct: float) -> str:
    """Return 'SAMPLE 0.01' or '' if sampling is disabled."""
    if sample_pct > 0:
        return f"SAMPLE {sample_pct}"
    return ""


def _recommend_column(client, db, table, col_name, col_type, sample_pct):
    """Returns (recommended_type, codec, reason)."""
    base = col_type.replace("Nullable(", "").rstrip(")")
    is_nullable = "Nullable(" in col_type

    # ── DateTime64 (sentinel dates like 0001-01-01) ───────
    if "DateTime64" in base:
        try:
            rows = q(client, f"""
                SELECT
                    countIf(toYear(toDateTime64(`{col_name}`, 6)) < 1900) AS pre_1900,
                    countIf(
                        toHour(toDateTime64(`{col_name}`, 6)) != 0
                        OR toMinute(toDateTime64(`{col_name}`, 6)) != 0
                        OR toSecond(toDateTime64(`{col_name}`, 6)) != 0
                    ) AS has_time,
                    count() AS cnt
                FROM `{db}`.`{table}` {_sample_clause(sample_pct)}
            """)
            r = rows[0]
            codec = "CODEC(DoubleDelta, ZSTD(1))"

            if r["pre_1900"] and r["pre_1900"] > 0:
                reason = (f"{r['pre_1900']} pre-1900 sentinels. "
                          f"Keep DateTime64(6). Consider converting 0001-01-01 → NULL.")
                return col_type, codec, reason
            elif r["has_time"] == 0:
                rec = "Nullable(Date32)" if is_nullable else "Date32"
                return rec, codec, "No time/pre-1900 → Date32 (4B vs 8B)"
            else:
                rec = "Nullable(DateTime)" if is_nullable else "DateTime"
                return rec, codec, "No sub-second/pre-1900 → DateTime (4B vs 8B)"
        except Exception as e:
            return col_type, "CODEC(DoubleDelta, ZSTD(1))", f"Error: {e}"

    if base in ("Date", "Date32", "DateTime"):
        return col_type, "CODEC(DoubleDelta, ZSTD(1))", "Already optimal"

    # ── Numeric (Decimal/Float/Int) ───────────────────────
    if any(k in base for k in ("Decimal", "Float")) or re.match(r'^U?Int\d+$', base):
        try:
            rows = q(client, f"""
                SELECT
                    min(toFloat64OrNull(toString(`{col_name}`)))  AS mn,
                    max(toFloat64OrNull(toString(`{col_name}`)))  AS mx,
                    max(CASE
                        WHEN position('.', toString(`{col_name}`)) > 0
                             AND toFloat64OrNull(toString(`{col_name}`))
                                 != floor(toFloat64OrNull(toString(`{col_name}`)))
                        THEN 1 ELSE 0
                    END) AS has_frac,
                    maxIf(
                        length(splitByChar('.', toString(`{col_name}`))[2]),
                        position('.', toString(`{col_name}`)) > 0
                    ) AS max_scale,
                    round(countIf(`{col_name}` IS NULL) * 100.0 / count(), 2) AS null_pct,
                    uniqHLL12(`{col_name}`) AS ndist
                FROM `{db}`.`{table}` {_sample_clause(sample_pct)}
            """)
        except Exception as e:
            return col_type, "CODEC(ZSTD(1))", f"Sampling error: {e}"

        r = rows[0]
        mn = r["mn"] or 0
        mx = r["mx"] or 0
        has_frac = r["has_frac"] or 0
        max_scale = int(r["max_scale"] or 0)
        null_pct = r["null_pct"] or 0
        ndist = r["ndist"] or 0

        if null_pct == 100:
            return "UInt8", "CODEC(ZSTD(1))", "100% NULL — consider dropping"

        if has_frac == 0:
            if mn >= 0:
                if mx < 256:            rec = "UInt8"
                elif mx < 65536:        rec = "UInt16"
                elif mx < 4294967296:   rec = "UInt32"
                else:                   rec = "UInt64"
            else:
                if mn >= -128 and mx < 128:                 rec = "Int8"
                elif mn >= -32768 and mx < 32768:           rec = "Int16"
                elif mn >= -2147483648 and mx < 2147483648:  rec = "Int32"
                else:                                        rec = "Int64"
            reason = f"Integer [{mn:.0f}..{mx:.0f}], {ndist} distinct"
            codec = "CODEC(Delta, ZSTD(1))"

            if ndist < 10_000 and rec not in ("UInt8", "Int8"):
                rec = f"LowCardinality({rec})"
                reason += " | LC wrapped"
        else:
            max_abs = max(abs(mn), abs(mx))
            s = max_scale
            if s <= 2 and max_abs < 1e9:      rec = f"Decimal32({s})"
            elif s <= 4 and max_abs < 1e18:   rec = f"Decimal64({s})"
            elif s <= 18 and max_abs < 1e38:  rec = f"Decimal128({s})"
            else:                              rec = f"Decimal256({s})"
            reason = f"Fractional scale={s}, range [{mn}..{mx}]"
            codec = "CODEC(Gorilla, ZSTD(1))"

        if is_nullable and 0 < null_pct < 100:
            rec = f"Nullable({rec})"

        return rec, codec, reason

    # ── String ────────────────────────────────────────────
    if "String" in base or "FixedString" in base:
        try:
            rows = q(client, f"""
                SELECT
                    uniqHLL12(`{col_name}`) AS ndist,
                    min(length(toString(`{col_name}`))) AS min_len,
                    max(length(toString(`{col_name}`))) AS max_len
                FROM `{db}`.`{table}` {_sample_clause(sample_pct)}
            """)
            r = rows[0]
            ndist = r["ndist"]
            reasons = []

            if "LowCardinality" in col_type:
                rec = col_type
                reasons.append("Already LowCardinality")
            elif ndist < 10_000:
                inner = col_type.replace("Nullable(", "").rstrip(")")
                if is_nullable:
                    rec = f"LowCardinality(Nullable({inner}))"
                else:
                    rec = f"LowCardinality({inner})"
                reasons.append(f"{ndist} distinct → LowCardinality")
            else:
                rec = col_type
                reasons.append(f"High cardinality ({ndist})")

            if r["min_len"] == r["max_len"] and 0 < (r["max_len"] or 0) <= 32:
                reasons.append(f"Fixed length {r['max_len']} → consider FixedString")

            return rec, "CODEC(ZSTD(1))", "; ".join(reasons)
        except Exception as e:
            return col_type, "CODEC(ZSTD(1))", f"Error: {e}"

    return col_type, "CODEC(ZSTD(1))", "No optimization identified"


# ═══════════════════════════════════════════════════════════════════════════
# DDL generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_local_v2_ddl(topo: TableTopology) -> str:
    """CREATE TABLE for the optimized local_v2, ON CLUSTER."""
    col_lines = []
    for c in topo.columns:
        col_lines.append(f"    `{c.name}` {c.recommended_type} {c.codec}")
    col_block = ",\n".join(col_lines)

    # Adjust ReplicatedMergeTree zoo path to avoid conflict with original
    engine = topo.engine_full
    if "Replicated" in engine:
        engine = re.sub(
            r"(ReplicatedMergeTree\(\s*'[^']*?)(')",
            rf"\1_v2\2",
            engine, count=1,
        )

    part_clause = f"\nPARTITION BY {topo.partition_by}" if topo.partition_by else ""

    return f"""CREATE TABLE IF NOT EXISTS `{topo.database}`.`{topo.local_v2}` ON CLUSTER `{topo.cluster}`
(
{col_block}
)
{engine}{part_clause}
ORDER BY ({topo.order_by})
{topo.settings_block};
"""


def generate_temp_dist_v2_ddl(topo: TableTopology) -> str:
    """Temp Distributed table pointing to local_v2 (used for fact migration)."""
    return f"""CREATE TABLE IF NOT EXISTS `{topo.database}`.`{topo.dist_v2}` ON CLUSTER `{topo.cluster}`
AS `{topo.database}`.`{topo.local_v2}`
ENGINE = Distributed('{topo.cluster}', '{topo.database}', '{topo.local_v2}', {topo.sharding_key});
"""


def generate_final_dist_ddl(topo: TableTopology) -> str:
    """Recreate the original Distributed table pointing to the (now-renamed) local."""
    if not topo.distributed_table:
        return ""
    return f"""CREATE TABLE `{topo.database}`.`{topo.distributed_table}` ON CLUSTER `{topo.cluster}`
AS `{topo.database}`.`{topo.local_table}`
ENGINE = Distributed('{topo.cluster}', '{topo.database}', '{topo.local_table}', {topo.sharding_key});
"""


# ═══════════════════════════════════════════════════════════════════════════
# Cast expression builder
# ═══════════════════════════════════════════════════════════════════════════
def generate_cast_select(columns: list[ColumnRec]) -> str:
    parts = []
    for c in columns:
        if not c.changed:
            parts.append(f"    `{c.name}`")
            continue

        target = c.recommended_type
        inner = target
        lc_wrap = nullable_wrap = False

        if inner.startswith("LowCardinality("):
            lc_wrap = True
            inner = inner[len("LowCardinality("):-1]
        if inner.startswith("Nullable("):
            nullable_wrap = True
            inner = inner[len("Nullable("):-1]

        expr = _cast_expr(c.name, inner, c.current_type)
        if nullable_wrap:
            expr = f"toNullable({expr})"
        if lc_wrap:
            expr = f"toLowCardinality({expr})"
        parts.append(f"    {expr} AS `{c.name}`")

    return ",\n".join(parts)


def _cast_expr(col: str, target: str, source: str) -> str:
    ints = {
        "UInt8": "toUInt8", "UInt16": "toUInt16",
        "UInt32": "toUInt32", "UInt64": "toUInt64",
        "Int8": "toInt8", "Int16": "toInt16",
        "Int32": "toInt32", "Int64": "toInt64",
    }
    if target in ints:
        if "Decimal" in source:
            return f"{ints[target]}(toFloat64(toString(`{col}`)))"
        return f"{ints[target]}(`{col}`)"

    m = re.match(r"Decimal(\d+)\((\d+)\)", target)
    if m:
        bits, scale = m.group(1), m.group(2)
        if "Decimal256" in source:
            return f"toDecimal{bits}(toFloat64(toString(`{col}`)), {scale})"
        return f"toDecimal{bits}(`{col}`, {scale})"

    if target == "Date32":   return f"toDate32(`{col}`)"
    if target == "Date":     return f"toDate(`{col}`)"
    if target == "DateTime": return f"toDateTime(`{col}`)"
    if "DateTime64" in target: return f"`{col}`"

    return f"CAST(`{col}` AS {target})"


# ═══════════════════════════════════════════════════════════════════════════
# Partition helpers
# ═══════════════════════════════════════════════════════════════════════════
def get_partitions(client, topo: TableTopology) -> list[dict]:
    """Get partition list with row counts."""
    if topo.table_type == "fact":
        # Aggregate across shards
        try:
            return q(client, f"""
                SELECT
                    partition_id,
                    sum(rows) AS total_rows
                FROM clusterAllReplicas('{topo.cluster}', system.parts)
                WHERE database = '{topo.database}'
                  AND table = '{topo.local_table}'
                  AND active = 1
                GROUP BY partition_id
                ORDER BY partition_id
            """)
        except Exception:
            pass

    # Dims: local parts (all nodes have same data)
    return q(client, f"""
        SELECT
            partition_id,
            sum(rows) AS total_rows
        FROM system.parts
        WHERE database = '{topo.database}'
          AND table = '{topo.local_table}'
          AND active = 1
        GROUP BY partition_id
        ORDER BY partition_id
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Replication lag checker (dims only)
# ═══════════════════════════════════════════════════════════════════════════
def wait_for_replication(client, topo: TableTopology, timeout: int = 600):
    """
    For dim tables: after INSERT on one node, wait for ZK to replicate
    to all other nodes before proceeding with RENAME.
    """
    if topo.table_type != "dim":
        return True

    logger.info(f"  Waiting for ZK replication of {topo.local_v2} ...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            lag_rows = q(client, f"""
                SELECT
                    host_name,
                    log_max_index - log_pointer AS queue_size
                FROM clusterAllReplicas('{topo.cluster}', system.replicas)
                WHERE database = '{topo.database}'
                  AND table = '{topo.local_v2}'
            """)

            if not lag_rows:
                time.sleep(5)
                continue

            max_lag = max(r["queue_size"] for r in lag_rows)
            if max_lag == 0:
                logger.info(f"  ✓ All replicas in sync (took {time.time()-start:.0f}s)")
                return True

            logger.info(f"  Replication queue: max {max_lag} entries pending, waiting...")
            time.sleep(10)

        except Exception as e:
            logger.warning(f"  Replication check error: {e}, retrying...")
            time.sleep(10)

    logger.error(f"  ⚠ Replication not complete after {timeout}s")
    logger.error(f"  Proceeding — verify manually before dropping _old table")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _step(n: int, msg: str):
    logger.info(f"\n{'━'*60}")
    logger.info(f"Step {n}: {msg}")
    logger.info(f"{'━'*60}")


def _run_or_log(client, sql: str, dry_run: bool):
    if dry_run:
        logger.info(f"  [DRY RUN] {sql[:300]}...")
    else:
        cmd(client, sql)
        logger.info(f"  ✓ Done")


def show_size_comparison(client, topo: TableTopology):
    """Print before/after disk size."""
    try:
        old_sz = q(client, f"""
            SELECT formatReadableSize(sum(bytes_on_disk)) AS sz
            FROM clusterAllReplicas('{topo.cluster}', system.parts)
            WHERE database = '{topo.database}'
              AND table = '{topo.local_old}' AND active = 1
        """)
        new_sz = q(client, f"""
            SELECT formatReadableSize(sum(bytes_on_disk)) AS sz
            FROM clusterAllReplicas('{topo.cluster}', system.parts)
            WHERE database = '{topo.database}'
              AND table = '{topo.local_table}' AND active = 1
        """)
        if old_sz and new_sz:
            logger.info(f"   Size: {old_sz[0]['sz']} → {new_sz[0]['sz']}")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# FACT migration
# ═══════════════════════════════════════════════════════════════════════════
def migrate_fact(client, topo: TableTopology, dry_run: bool, pause: int) -> bool:
    """
    Fact table migration via parallel_distributed_insert_select=2.

    Flow:
      INSERT INTO dist_v2 SELECT FROM dist
      → coordinator tells each shard to read its own local + write its own local_v2
      → zero cross-shard traffic
    """
    db = topo.database

    migration_settings = {
        "parallel_distributed_insert_select": 2,
        "insert_distributed_sync": 1,
        "max_insert_threads": 4,
        "max_execution_time": 7200,
        "send_timeout": 3600,
        "receive_timeout": 3600,
    }

    # 1. CREATE local_v2 ON CLUSTER
    _step(1, f"CREATE {topo.local_v2} ON CLUSTER")
    _run_or_log(client, generate_local_v2_ddl(topo), dry_run)

    # 2. CREATE temp Distributed → local_v2
    _step(2, f"CREATE temp Distributed {topo.dist_v2}")
    _run_or_log(client, generate_temp_dist_v2_ddl(topo), dry_run)

    # 3. INSERT partition by partition
    _step(3, "INSERT via parallel_distributed_insert_select=2 (shard-local)")
    cast_select = generate_cast_select(topo.columns)
    partitions = get_partitions(client, topo)
    source = topo.distributed_table

    if not partitions:
        sql = f"INSERT INTO `{db}`.`{topo.dist_v2}` SELECT\n{cast_select}\nFROM `{db}`.`{source}`"
        if dry_run:
            logger.info(f"  [DRY RUN] Single INSERT (no partitions)")
        else:
            cmd(client, sql, settings=migration_settings)
            logger.info(f"  ✓ Full table migrated")
    else:
        total_rows = sum(p["total_rows"] for p in partitions)
        logger.info(f"  {len(partitions)} partitions, ~{total_rows:,} rows")
        migrated = 0

        for i, part in enumerate(partitions):
            pid, prows = part["partition_id"], part["total_rows"]
            sql = (f"INSERT INTO `{db}`.`{topo.dist_v2}` SELECT\n{cast_select}\n"
                   f"FROM `{db}`.`{source}` WHERE _partition_id = '{pid}'")

            if dry_run:
                if i == 0:
                    logger.info(f"  [DRY RUN] Sample:\n{sql[:400]}...")
                logger.info(f"  [DRY RUN] {pid} (~{prows:,} rows)")
            else:
                t0 = time.time()
                try:
                    cmd(client, sql, settings=migration_settings)
                    migrated += prows
                    pct = round(migrated / total_rows * 100, 1) if total_rows else 0
                    logger.info(f"  ✓ {pid} | {prows:>12,} rows | "
                                f"{time.time()-t0:.1f}s | {pct}%")
                except Exception as e:
                    logger.error(f"  ✗ {pid} FAILED: {e}")
                    logger.error(f"  Retry: ALTER TABLE `{db}`.`{topo.local_v2}` "
                                 f"ON CLUSTER `{topo.cluster}` DROP PARTITION '{pid}';")
                    return False

                if pause and i < len(partitions) - 1:
                    time.sleep(pause)

    # 4. Validate
    _step(4, "Validate row counts")
    if not dry_run:
        old_cnt = q(client, f"SELECT count() AS c FROM `{db}`.`{source}`")[0]["c"]
        new_cnt = q(client, f"SELECT count() AS c FROM `{db}`.`{topo.dist_v2}`")[0]["c"]
        if old_cnt != new_cnt:
            logger.error(f"  ❌ Mismatch! old={old_cnt:,} new={new_cnt:,}")
            return False
        logger.info(f"  ✓ Match: {old_cnt:,}")

        # Per-shard check
        try:
            sc = q(client, f"""
                SELECT _shard_num,
                    countIf(table = '{topo.local_table}') AS old_r,
                    countIf(table = '{topo.local_v2}') AS new_r
                FROM clusterAllReplicas('{topo.cluster}', system.parts)
                WHERE database = '{db}'
                  AND table IN ('{topo.local_table}', '{topo.local_v2}')
                  AND active = 1
                GROUP BY _shard_num ORDER BY _shard_num
            """)
            for s in sc:
                ok = "✓" if s["old_r"] == s["new_r"] else "✗"
                logger.info(f"    Shard {s['_shard_num']}: old={s['old_r']:,} new={s['new_r']:,} {ok}")
                if s["old_r"] != s["new_r"]:
                    logger.error("  ❌ Per-shard mismatch!")
                    return False
        except Exception as e:
            logger.warning(f"  Per-shard check skipped: {e}")

    # 5. DROP temp dist_v2
    _step(5, f"DROP temp {topo.dist_v2}")
    _run_or_log(client,
                f"DROP TABLE IF EXISTS `{db}`.`{topo.dist_v2}` ON CLUSTER `{topo.cluster}`",
                dry_run)

    # 6. DROP original Distributed
    if topo.distributed_table:
        _step(6, f"DROP Distributed {topo.distributed_table}")
        _run_or_log(client,
                    f"DROP TABLE IF EXISTS `{db}`.`{topo.distributed_table}` "
                    f"ON CLUSTER `{topo.cluster}`",
                    dry_run)

    # 7. RENAME local ON CLUSTER
    _step(7, "Atomic RENAME")
    _run_or_log(client,
                f"RENAME TABLE "
                f"`{db}`.`{topo.local_table}` TO `{db}`.`{topo.local_old}`, "
                f"`{db}`.`{topo.local_v2}` TO `{db}`.`{topo.local_table}` "
                f"ON CLUSTER `{topo.cluster}`",
                dry_run)

    # 8. Recreate Distributed
    if topo.distributed_table:
        _step(8, f"Recreate Distributed {topo.distributed_table}")
        _run_or_log(client, generate_final_dist_ddl(topo), dry_run)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# DIM migration
# ═══════════════════════════════════════════════════════════════════════════
def migrate_dim(client, topo: TableTopology, dry_run: bool, pause: int,
                repl_timeout: int) -> bool:
    """
    Dim table migration: local INSERT on one node, ZK replicates to 5 others.

    Key differences from facts:
      - INSERT into LOCAL _v2 table directly (NOT through Distributed)
      - One node does the work, ZK handles the rest
      - Must wait for replication before RENAME
      - NEVER set parallel_distributed_insert_select (would cause 6x writes)
    """
    db = topo.database

    dim_settings = {
        "max_insert_threads": 4,
        "max_execution_time": 7200,
        "send_timeout": 3600,
        "receive_timeout": 3600,
        # ⚠ NO parallel_distributed_insert_select here
    }

    # 1. CREATE local_v2 ON CLUSTER
    _step(1, f"CREATE {topo.local_v2} ON CLUSTER")
    _run_or_log(client, generate_local_v2_ddl(topo), dry_run)

    # 2. INSERT partition by partition (local → local_v2, one node)
    _step(2, "INSERT local → local_v2 (one node, ZK replicates)")
    cast_select = generate_cast_select(topo.columns)
    partitions = get_partitions(client, topo)

    if not partitions:
        sql = (f"INSERT INTO `{db}`.`{topo.local_v2}` SELECT\n{cast_select}\n"
               f"FROM `{db}`.`{topo.local_table}`")
        if dry_run:
            logger.info(f"  [DRY RUN] Single INSERT (no partitions)")
        else:
            cmd(client, sql, settings=dim_settings)
            logger.info(f"  ✓ Full table migrated on this node")
    else:
        total_rows = sum(p["total_rows"] for p in partitions)
        logger.info(f"  {len(partitions)} partitions, ~{total_rows:,} rows")
        logger.info(f"  INSERT on this node → ZK replicates to 5 others")
        migrated = 0

        for i, part in enumerate(partitions):
            pid, prows = part["partition_id"], part["total_rows"]

            # INSERT into LOCAL v2 — NOT Distributed!
            sql = (f"INSERT INTO `{db}`.`{topo.local_v2}` SELECT\n{cast_select}\n"
                   f"FROM `{db}`.`{topo.local_table}` WHERE _partition_id = '{pid}'")

            if dry_run:
                if i == 0:
                    logger.info(f"  [DRY RUN] Sample:\n{sql[:400]}...")
                logger.info(f"  [DRY RUN] {pid} (~{prows:,} rows)")
            else:
                t0 = time.time()
                try:
                    cmd(client, sql, settings=dim_settings)
                    migrated += prows
                    pct = round(migrated / total_rows * 100, 1) if total_rows else 0
                    logger.info(f"  ✓ {pid} | {prows:>12,} rows | "
                                f"{time.time()-t0:.1f}s | {pct}%")
                except Exception as e:
                    logger.error(f"  ✗ {pid} FAILED: {e}")
                    logger.error(f"  Retry: ALTER TABLE `{db}`.`{topo.local_v2}` "
                                 f"ON CLUSTER `{topo.cluster}` DROP PARTITION '{pid}';")
                    return False

                if pause and i < len(partitions) - 1:
                    time.sleep(pause)

    # 3. Wait for ZK replication
    _step(3, "Wait for ZK replication to all 6 nodes")
    if not dry_run:
        wait_for_replication(client, topo, timeout=repl_timeout)
    else:
        logger.info(f"  [DRY RUN] Would wait up to {repl_timeout}s")

    # 4. Validate
    _step(4, "Validate row counts")
    if not dry_run:
        old_cnt = q(client, f"SELECT count() AS c FROM `{db}`.`{topo.local_table}`")[0]["c"]
        new_cnt = q(client, f"SELECT count() AS c FROM `{db}`.`{topo.local_v2}`")[0]["c"]
        if old_cnt != new_cnt:
            logger.error(f"  ❌ Mismatch! old={old_cnt:,} new={new_cnt:,}")
            return False
        logger.info(f"  ✓ Match on this node: {old_cnt:,}")

        # Cross-node check
        try:
            nc = q(client, f"""
                SELECT host_name,
                    countIf(table = '{topo.local_v2}') AS v2_rows
                FROM clusterAllReplicas('{topo.cluster}', system.parts)
                WHERE database = '{db}'
                  AND table = '{topo.local_v2}' AND active = 1
                GROUP BY host_name ORDER BY host_name
            """)
            for n in nc:
                ok = "✓" if n["v2_rows"] == old_cnt else "⚠ replication lag?"
                logger.info(f"    {n['host_name']}: {n['v2_rows']:,} {ok}")
        except Exception as e:
            logger.warning(f"  Cross-node check skipped: {e}")

    # 5. DROP Distributed
    if topo.distributed_table:
        _step(5, f"DROP Distributed {topo.distributed_table}")
        _run_or_log(client,
                    f"DROP TABLE IF EXISTS `{db}`.`{topo.distributed_table}` "
                    f"ON CLUSTER `{topo.cluster}`",
                    dry_run)

    # 6. RENAME local ON CLUSTER
    _step(6, "Atomic RENAME")
    _run_or_log(client,
                f"RENAME TABLE "
                f"`{db}`.`{topo.local_table}` TO `{db}`.`{topo.local_old}`, "
                f"`{db}`.`{topo.local_v2}` TO `{db}`.`{topo.local_table}` "
                f"ON CLUSTER `{topo.cluster}`",
                dry_run)

    # 7. Recreate Distributed
    if topo.distributed_table:
        _step(7, f"Recreate Distributed {topo.distributed_table}")
        _run_or_log(client, generate_final_dist_ddl(topo), dry_run)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Table discovery for --all-decimal256
# ═══════════════════════════════════════════════════════════════════════════
def find_decimal256_tables(client, db: str, table_type: str) -> list[str]:
    """Find tables with Decimal256 matching the naming convention."""
    if table_type == "fact":
        # Fact local tables end with _local
        rows = q(client, f"""
            SELECT DISTINCT table
            FROM system.columns
            WHERE database = '{db}'
              AND type LIKE '%Decimal256%'
              AND table LIKE '%\\_local'
              AND table NOT LIKE '%\\_v2%'
              AND table NOT LIKE '%\\_old%'
        """)
        # Return base name (dist name = local minus _local)
        return [r["table"][:-6] for r in rows]
    else:
        # Dim local tables: NOT ending with _local, _dist, _v2, _old
        rows = q(client, f"""
            SELECT DISTINCT table
            FROM system.columns
            WHERE database = '{db}'
              AND type LIKE '%Decimal256%'
              AND table NOT LIKE '%\\_local'
              AND table NOT LIKE '%\\_dist'
              AND table NOT LIKE '%\\_v2%'
              AND table NOT LIKE '%\\_old%'
        """)
        return [r["table"] for r in rows]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CH Migration Automator (LB Compatible, Fact + Dim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Naming conventions:
  fact:  local = {name}_local    dist = {name}
  dim:   local = {name}          dist = {name}_dist

Examples:
  # Fact table dry run
  %(prog)s --host lb --database db --cluster c \\
      --table app_fact_trades --table-type fact --dry-run

  # Dim table migration
  %(prog)s --host lb --database db --cluster c \\
      --table app_ref_currency --table-type dim

  # All Decimal256 facts
  %(prog)s --host lb --database db --cluster c \\
      --all-decimal256 --table-type fact --dry-run
        """,
    )
    parser.add_argument("--host",          required=True, help="LB hostname")
    parser.add_argument("--port",          type=int, default=None)
    parser.add_argument("--user",          default="default")
    parser.add_argument("--password",      default="")
    parser.add_argument("--database",      required=True)
    parser.add_argument("--cluster",       required=True)
    parser.add_argument("--table-type",    required=True, choices=["fact", "dim"],
                        help="fact = sharded | dim = replicated full-copy")
    parser.add_argument("--table",         default=None, help="Single table")
    parser.add_argument("--tables",        nargs="*", default=None)
    parser.add_argument("--all-decimal256", action="store_true")
    parser.add_argument("--sample-pct",    type=float, default=0.01,
                        help="Profiling sample (0.01 = 1%%)")
    parser.add_argument("--no-sample",     action="store_true",
                        help="Disable SAMPLE clause — scan full dataset for profiling "
                             "(required if your CH engine/table doesn't support SAMPLE)")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--pause-between-partitions", type=int, default=10)
    parser.add_argument("--replication-timeout", type=int, default=600,
                        help="Max wait for dim ZK replication (default 600s)")
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    # --no-sample overrides --sample-pct
    if args.no_sample:
        args.sample_pct = 0

    client = make_client(args.host, args.port, args.user, args.password,
                         args.database)

    # Determine table list
    if args.table:
        table_list = [args.table]
    elif args.tables:
        table_list = args.tables
    elif args.all_decimal256:
        table_list = find_decimal256_tables(client, args.database, args.table_type)
        logger.info(f"Found {len(table_list)} {args.table_type} tables with Decimal256:")
        for t in table_list:
            logger.info(f"  • {t}")
    else:
        parser.error("Specify --table, --tables, or --all-decimal256")
        return

    if not table_list:
        logger.info("No tables to process.")
        return

    all_reports = []
    results = []

    for tbl in table_list:
        base, local, dist = resolve_names(tbl, args.table_type)

        logger.info(f"\n{'═'*60}")
        logger.info(f"  {args.table_type.upper()}: {tbl}")
        logger.info(f"  Local: {local}  |  Dist: {dist}")
        logger.info(f"{'═'*60}")

        try:
            topo = discover_topology(client, args.database, tbl,
                                     args.table_type, args.cluster)
        except Exception as e:
            logger.error(f"  Topology error: {e}")
            results.append((tbl, "SKIP", str(e)))
            continue

        logger.info(f"  ORDER BY:  {topo.order_by}")
        logger.info(f"  PARTITION: {topo.partition_by or '(none)'}")
        if args.table_type == "fact":
            logger.info(f"  Sharding:  {topo.sharding_key}")
        else:
            logger.info(f"  Replication: full copy on all 6 nodes")

        # Profile
        sample_msg = "full scan" if args.sample_pct == 0 else f"{args.sample_pct*100:.1f}% sample"
        logger.info(f"\n  Profiling ({sample_msg})...")
        profile_columns(client, topo, args.sample_pct)

        changes = [c for c in topo.columns if c.changed]
        if not changes:
            logger.info(f"  ✓ Already optimal. Skipping.")
            results.append((tbl, "SKIP", "already optimal"))
            continue

        logger.info(f"\n  {len(changes)} columns to optimize:")
        for c in changes:
            flags = ""
            if c.is_in_order_by:     flags += " ⚠ ORDER BY"
            if c.is_in_partition_by: flags += " ⚠ PARTITION BY"
            logger.info(f"    {c.name}: {c.current_type} → {c.recommended_type}"
                        f"  [{c.compressed_mb} MB]{flags}")
            logger.info(f"      {c.reason}")

        all_reports.append({
            "table": tbl,
            "type": args.table_type,
            "local_table": topo.local_table,
            "distributed_table": topo.distributed_table,
            "columns": [asdict(c) for c in topo.columns],
        })

        # Migrate
        if args.table_type == "fact":
            ok = migrate_fact(client, topo, args.dry_run,
                              args.pause_between_partitions)
        else:
            ok = migrate_dim(client, topo, args.dry_run,
                             args.pause_between_partitions,
                             args.replication_timeout)

        if ok:
            if not args.dry_run:
                show_size_comparison(client, topo)
            results.append((tbl, "OK" if not args.dry_run else "DRY RUN", ""))
        else:
            results.append((tbl, "FAILED", "see logs"))
            if not args.dry_run:
                logger.error(f"Stopping on failure: {tbl}")
                break

    # Summary
    logger.info(f"\n{'═'*60}")
    logger.info(f"  SUMMARY ({args.table_type.upper()} tables)")
    logger.info(f"{'═'*60}")
    for tbl, status, note in results:
        extra = f" ({note})" if note else ""
        logger.info(f"  {status:<10} {tbl}{extra}")

    if not args.dry_run:
        ok_tables = [tbl for tbl, s, _ in results if s == "OK"]
        if ok_tables:
            logger.info(f"\n  Cleanup (after verification):")
            for tbl in ok_tables:
                _, local, _ = resolve_names(tbl, args.table_type)
                logger.info(f"    DROP TABLE `{args.database}`.`{local}_old` "
                             f"ON CLUSTER `{args.cluster}`;")

    if args.output_report and all_reports:
        with open(args.output_report, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        logger.info(f"\n  Report: {args.output_report}")


if __name__ == "__main__":
    main()