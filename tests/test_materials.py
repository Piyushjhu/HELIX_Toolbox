"""Tests for helix.materials."""
import pytest
from helix.materials.database import get_material_properties, list_available_materials
from helix.materials.resolver import MaterialResolver


class TestDatabase:
    def test_exact_match(self):
        props = get_material_properties("Copper")
        assert props["material_found"] is True
        assert props["density"] == 8960

    def test_case_insensitive(self):
        props = get_material_properties("copper")
        assert props["material_found"] is True

    def test_unknown_material(self):
        props = get_material_properties("Unobtanium")
        assert props["material_found"] is False
        assert props["density"] == 8960  # default

    def test_custom_defaults(self):
        props = get_material_properties("Unobtanium", default_density=1000)
        assert props["density"] == 1000

    def test_list_materials(self):
        mats = list_available_materials()
        assert "Cu" in mats
        assert "Al" in mats


class TestResolver:
    def test_config_override(self):
        resolver = MaterialResolver(
            config_materials={"MyAlloy": {"density": 5000, "bulk_wave_speed": 4000, "C_L": 4500}}
        )
        props = resolver.resolve("MyAlloy")
        assert props["density"] == 5000
        assert props["C_L"] == 4500
        assert props["source"] == "config"

    def test_database_fallback(self):
        resolver = MaterialResolver()
        props = resolver.resolve("Copper")
        assert props["source"] == "database"
        assert props["density"] == 8960

    def test_unknown_default(self):
        resolver = MaterialResolver(default_density=7777)
        props = resolver.resolve("Unobtanium")
        assert props["source"] == "default"
        assert props["density"] == 7777

    def test_case_insensitive_config(self):
        resolver = MaterialResolver(config_materials={"Cu": {"density": 9000, "C0": 4000}})
        props = resolver.resolve("cu")
        assert props["density"] == 9000
        assert props["source"] == "config"
