"""
Tests for ClickHouse service: SQL safety validation, query execution, schema introspection.
"""

import pytest
from config import Settings


@pytest.fixture
def ch_settings():
    return Settings(
        llm_api_key="test",
        ch_host="localhost",
        ch_database="trading_db",
        allowed_ch_databases=["trading_db"],
        sql_blocked_keywords=[
            "DROP", "TRUNCATE", "ALTER", "CREATE", "INSERT",
            "UPDATE", "DELETE", "GRANT", "REVOKE", "ATTACH",
            "DETACH", "RENAME", "OPTIMIZE", "KILL",
        ],
    )


@pytest.fixture
def ch_service(ch_settings):
    from services.clickhouse_service import ClickHouseService
    return ClickHouseService(ch_settings)


# ── SQL Safety Tests ─────────────────────────────────────────────


class TestSQLSafety:
    """Defense-in-depth: programmatic SQL validation."""

    def test_safe_select(self, ch_service):
        sql = "SELECT desk, SUM(pnl) FROM trades GROUP BY desk ORDER BY 2 DESC LIMIT 10"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True
        assert issues == []

    def test_safe_with_cte(self, ch_service):
        sql = """WITH daily AS (
            SELECT trade_date, SUM(pnl) as daily_pnl
            FROM trades GROUP BY trade_date
        )
        SELECT * FROM daily ORDER BY trade_date LIMIT 100"""
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True

    def test_safe_explain(self, ch_service):
        sql = "EXPLAIN SELECT * FROM trades LIMIT 10"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True

    def test_safe_describe(self, ch_service):
        sql = "DESCRIBE TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True

    def test_block_drop(self, ch_service):
        sql = "DROP TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False
        assert any("DROP" in i for i in issues)

    def test_block_insert(self, ch_service):
        sql = "INSERT INTO trades VALUES (1, '2024-01-01', 'AAPL', 100)"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False
        assert any("INSERT" in i for i in issues)

    def test_block_delete(self, ch_service):
        sql = "DELETE FROM trades WHERE trade_id = 1"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_alter(self, ch_service):
        sql = "ALTER TABLE trades ADD COLUMN new_col String"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_truncate(self, ch_service):
        sql = "TRUNCATE TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_grant(self, ch_service):
        sql = "GRANT SELECT ON trades TO user1"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_semicolon_injection(self, ch_service):
        sql = "SELECT * FROM trades LIMIT 10; DROP TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False
        assert any("Semicolon" in i or "multi-statement" in i.lower() for i in issues)

    def test_semicolon_in_string_literal_ok(self, ch_service):
        sql = "SELECT * FROM trades WHERE comment = 'test; value' LIMIT 10"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        # Semicolon inside string literal should be OK
        assert is_safe is True

    def test_require_limit_on_select(self, ch_service):
        sql = "SELECT * FROM trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False
        assert any("LIMIT" in i for i in issues)

    def test_block_system_table_write(self, ch_service):
        sql = "SELECT * FROM system.query_log LIMIT 10"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        # system.query_log is in the allowed list
        assert is_safe is True

    def test_block_restricted_system_table(self, ch_service):
        sql = "SELECT * FROM system.users LIMIT 10"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_case_insensitive_blocking(self, ch_service):
        sql = "drop table trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_optimize(self, ch_service):
        sql = "OPTIMIZE TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_kill(self, ch_service):
        sql = "KILL QUERY WHERE query_id = 'abc'"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_block_attach(self, ch_service):
        sql = "ATTACH TABLE trades"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is False

    def test_select_with_prewhere(self, ch_service):
        sql = """SELECT desk, SUM(pnl)
        FROM trades
        PREWHERE trade_date >= '2024-01-01'
        WHERE volume > 100
        GROUP BY desk
        LIMIT 100"""
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True

    def test_show_tables(self, ch_service):
        sql = "SHOW TABLES FROM trading_db"
        is_safe, issues = ch_service.validate_sql_safety(sql)
        assert is_safe is True


# ── Query Execution (mocked) ────────────────────────────────────


class TestQueryExecution:
    def test_blocked_query_returns_error(self, ch_service):
        result = ch_service.execute_query("DROP TABLE trades")
        assert result["success"] is False
        assert "safety check failed" in result["error"]

    def test_missing_limit_returns_error(self, ch_service):
        result = ch_service.execute_query("SELECT * FROM trades")
        assert result["success"] is False
