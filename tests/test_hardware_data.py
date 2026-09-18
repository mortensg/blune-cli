from blune_cli.hardware_data import ALL_CHIPS, BY_ID


def test_m4_pro_is_registered():
    """The gap that started this whole project -- must never regress."""
    assert "m4_pro" in BY_ID
    m4_pro = BY_ID["m4_pro"]
    assert m4_pro.name == "Apple M4 Pro"
    assert m4_pro.memory_bandwidth_gbs == 273
    assert m4_pro.max_unified_or_vram_gb == 48


def test_all_apple_silicon_tiers_present():
    for gen in ("m1", "m2", "m3", "m4"):
        assert gen in BY_ID
        for tier in ("pro", "max", "ultra"):
            assert f"{gen}_{tier}" in BY_ID


def test_chip_ids_are_unique():
    ids = [c.id for c in ALL_CHIPS]
    assert len(ids) == len(set(ids))


def test_all_chips_have_positive_bandwidth():
    for c in ALL_CHIPS:
        assert c.memory_bandwidth_gbs > 0
