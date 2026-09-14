# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Unit tests for stateless PG ``_world`` registration helpers.

These helpers are wired into ``NPUPlatform.on_stateless_process_group_created``
/ ``on_stateless_process_group_destroyed`` lifecycle hooks.
"""

import unittest
from unittest.mock import MagicMock

import vllm_ascend.distributed.stateless_coordinator as sc


class _FakePG:
    def __init__(self, group_name: str, size: int = 2) -> None:
        self.group_name = group_name
        self._size = size
        self.store = MagicMock()

    def size(self) -> int:
        return self._size

    def get_group_store(self):
        return self.store


class _StubWorld:
    def __init__(self) -> None:
        self.pg_group_ranks: dict = {}
        self.pg_map: dict = {}
        self.pg_names: dict = {}
        self.pg_backend_config: dict = {}
        self.default_pg = None


class TestRegisterPG(unittest.TestCase):
    def setUp(self):
        self.world = _StubWorld()
        sc._world = self.world
        self.addCleanup(setattr, sc, "_world", sc._world)

    def test_register_pg_writes_world_state(self):
        pg = _FakePG("mc2:0_device")
        sc._register_pg(pg, "hccl")
        self.assertEqual(self.world.pg_map[pg], ("hccl", pg.store))
        self.assertEqual(self.world.pg_names[pg], "mc2:0_device")
        self.assertEqual(self.world.pg_group_ranks[pg], {0: 0, 1: 1})
        self.assertIn("hccl", self.world.pg_backend_config[pg])
        self.assertIsNone(self.world.default_pg)

    def test_register_pg_does_not_set_default_pg_for_lowercase_world(self):
        # The original patch only matched uppercase "WORLD" in the group
        # name; upstream stateless names are lowercase, so default_pg is
        # left untouched.
        pg = _FakePG("world:0_device")
        sc._register_pg(pg, "hccl")
        self.assertIsNone(self.world.default_pg)

    def test_register_pg_sets_default_pg_for_world_group(self):
        pg = _FakePG("WORLD:0_device")
        sc._register_pg(pg, "hccl")
        self.assertIs(self.world.default_pg, pg)

    def test_unregister_pg_removes_world_state(self):
        pg = _FakePG("ep:0_cpu")
        sc._register_pg(pg, "gloo")
        sc._unregister_pg(pg)
        self.assertNotIn(pg, self.world.pg_map)
        self.assertNotIn(pg, self.world.pg_names)
        self.assertNotIn(pg, self.world.pg_group_ranks)
        self.assertNotIn(pg, self.world.pg_backend_config)

    def test_unregister_pg_tolerates_missing_entry(self):
        pg = _FakePG("dp:0_device")
        sc._unregister_pg(pg)
        self.assertIsNone(self.world.default_pg)

    def test_register_pg_accepts_none_group_name(self):
        pg = _FakePG(None)
        sc._register_pg(pg, "hccl")
        self.assertIsNone(self.world.pg_names[pg])
        self.assertIsNone(self.world.default_pg)


if __name__ == "__main__":
    unittest.main()
