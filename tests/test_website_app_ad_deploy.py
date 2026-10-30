"""Exercise the deployer's actual SQLite comparison with advertisement BLOBs."""
import ast
import hashlib
import json
import sqlite3
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "website/deploy/update.py"
TREE = ast.parse(SOURCE.read_text())
LOGICAL = next(node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == "logical")
UNCHANGED = next(node for node in ast.walk(TREE) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "unchanged" for target in node.targets))
NAMESPACE = {"hashlib": hashlib, "json": json}
exec(compile(ast.Module(body=[LOGICAL], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)


def unchanged(before, after):
    scope = {**NAMESPACE, "before": before, "after": after}
    exec(compile(ast.Module(body=[UNCHANGED], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["unchanged"]


class AppAdDeploymentTests(unittest.TestCase):
    def test_png_blob_comparison_is_stable_and_detects_changes(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE app_ad (id INTEGER PRIMARY KEY, image BLOB)")
            db.execute("INSERT INTO app_ad VALUES (1,?)", (b"\x89PNG\x00\xff",))
            first = NAMESPACE["logical"](db)
            self.assertTrue(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("UPDATE app_ad SET image=?", (b"different PNG",))
            self.assertFalse(unchanged(first, NAMESPACE["logical"](db)))

    def test_only_new_empty_ad_and_start_tables_are_permitted(self):
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY, name TEXT)")
            db.execute("INSERT INTO entries VALUES (1,'existing business row')")
            first = NAMESPACE["logical"](db)
            db.execute("CREATE TABLE app_ad (id INTEGER PRIMARY KEY, image BLOB)")
            self.assertTrue(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("INSERT INTO app_ad VALUES (1,?)", (b"unrequested ad",))
            self.assertFalse(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("DELETE FROM app_ad")
            for name in ("app_start_totals", "app_start_installs", "app_start_events", "app_start_daily"):
                db.execute(f"CREATE TABLE {name} (id TEXT PRIMARY KEY)")
            self.assertTrue(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("INSERT INTO app_start_events VALUES ('unrequested event')")
            self.assertFalse(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("DELETE FROM app_start_events")
            db.execute("UPDATE entries SET name='changed'")
            self.assertFalse(unchanged(first, NAMESPACE["logical"](db)))
            db.execute("UPDATE entries SET name='existing business row'")
            db.execute("CREATE TABLE unexpected (id INTEGER PRIMARY KEY)")
            self.assertFalse(unchanged(first, NAMESPACE["logical"](db)))


if __name__ == "__main__":
    unittest.main()
