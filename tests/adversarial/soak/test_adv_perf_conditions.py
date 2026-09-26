"""The machine a timing run describes, and when a run is invalid rather than failed.

A timing taken on battery or on a busy machine measures the machine, not memvara, so the
design marks such a run invalid rather than failed. The fingerprint names the machine
that measured a budget, so a budget is only ever compared with timings from the same
hardware.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import perf_budget as pb


def test_the_fingerprint_names_the_machine_and_the_software_it_ran() -> None:
    first, second = pb.machine_fingerprint(), pb.machine_fingerprint()
    assert set(first) == {"system", "release", "machine", "cpu", "logical_cpus",
                          "memory_bytes", "python", "sqlite", "memvara", "commit", "id"}
    assert first["id"] == second["id"]
    assert first["logical_cpus"] == os.cpu_count()


def test_a_python_upgrade_keeps_the_machine_but_other_hardware_does_not(
        monkeypatch: pytest.MonkeyPatch) -> None:
    before = pb.machine_fingerprint()["id"]
    monkeypatch.setattr(pb.platform, "python_version", lambda: "9.9.9")
    assert pb.machine_fingerprint()["id"] == before
    monkeypatch.setattr(pb.os, "cpu_count", lambda: 999)
    assert pb.machine_fingerprint()["id"] != before


def test_a_run_with_no_battery_and_little_load_is_valid() -> None:
    assert pb.invalid_reasons([pb.Conditions(False, 0.2), pb.Conditions(None, 0.5)]) == []


def test_a_run_on_battery_is_invalid() -> None:
    reasons = pb.invalid_reasons([pb.Conditions(True, 0.1), pb.Conditions(True, 0.1)])
    assert len(reasons) == 1 and "battery" in reasons[0] and "2 of 2" in reasons[0]


def test_a_run_under_load_is_invalid_and_says_how_busy_the_machine_was() -> None:
    reasons = pb.invalid_reasons([pb.Conditions(False, 0.3), pb.Conditions(False, 0.9)])
    assert len(reasons) == 1 and "0.90" in reasons[0] and "0.5" in reasons[0]


def test_battery_and_load_are_two_reasons() -> None:
    assert len(pb.invalid_reasons([pb.Conditions(True, 0.9)])) == 2


def test_the_conditions_of_this_machine_can_be_read() -> None:
    now = pb.read_conditions()
    assert isinstance(now.load_per_cpu, float) and now.load_per_cpu >= 0.0
    assert now.on_battery in (True, False, None)


@pytest.mark.parametrize("text, expected", [
    ("Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t71%; discharging", True),
    ("Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged", False),
    ("Now drawing from 'UPS Power'\n", None),
    ("", None),
])
def test_macos_power_is_read_from_pmset(text: str, expected: bool | None) -> None:
    assert pb.on_battery_from_pmset(text) is expected


def supply(root: pathlib.Path, name: str, kind: str, **files: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "type").write_text(kind + "\n", encoding="utf-8")
    for filename, value in files.items():
        (folder / filename).write_text(value + "\n", encoding="utf-8")


def test_linux_power_on_mains_is_not_battery(tmp_path: pathlib.Path) -> None:
    supply(tmp_path, "AC", "Mains", online="1")
    supply(tmp_path, "BAT0", "Battery", status="Charging")
    assert pb.on_battery_from_sysfs(tmp_path) is False


def test_linux_power_off_mains_with_a_battery_is_battery(tmp_path: pathlib.Path) -> None:
    supply(tmp_path, "AC", "Mains", online="0")
    supply(tmp_path, "BAT0", "Battery", status="Discharging")
    assert pb.on_battery_from_sysfs(tmp_path) is True


def test_linux_with_no_power_supply_is_unknown(tmp_path: pathlib.Path) -> None:
    assert pb.on_battery_from_sysfs(tmp_path) is None
    assert pb.on_battery_from_sysfs(tmp_path / "missing") is None
