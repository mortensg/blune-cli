import platform

import pytest

from blune_cli.hardware import detect_machine


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only detection path")
def test_detect_machine_on_macos_returns_sane_values():
    machine = detect_machine()
    assert machine.os_name == "macOS"
    assert machine.total_ram_gb > 0
    assert "Apple" in machine.chip_name or machine.chip_name == "unknown Apple Silicon"


def test_detected_machine_bandwidth_property_without_chip_spec():
    from blune_cli.hardware import DetectedMachine

    m = DetectedMachine(os_name="Linux", chip_name="unknown CPU", total_ram_gb=16, chip_spec=None)
    assert m.bandwidth_gbs is None
