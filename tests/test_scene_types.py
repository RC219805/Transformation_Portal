"""Tests for scene type taxonomy module.

Provides comprehensive test coverage for scene type normalization,
validation, and retrieval functions used across run cards, RAG retrieval,
and quality analysis.
"""

from __future__ import annotations

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.scene_types import (
    SCENE_TYPES,
    get_all_aliases,
    get_scene_type_description,
    list_scene_types,
    normalize_scene_type,
    validate_scene_type,
)


class TestSceneTypeConstants:
    """Tests for SCENE_TYPES constant structure."""

    def test_scene_types_is_dict(self):
        """SCENE_TYPES should be a dictionary."""
        assert isinstance(SCENE_TYPES, dict)

    def test_scene_types_not_empty(self):
        """SCENE_TYPES should contain entries."""
        assert len(SCENE_TYPES) > 0

    def test_all_entries_have_aliases(self):
        """Every scene type must have aliases list."""
        for scene_type, config in SCENE_TYPES.items():
            assert "aliases" in config, f"{scene_type} missing 'aliases'"
            assert isinstance(config["aliases"], list), f"{scene_type} aliases must be list"

    def test_all_entries_have_description(self):
        """Every scene type must have a description."""
        for scene_type, config in SCENE_TYPES.items():
            assert "description" in config, f"{scene_type} missing 'description'"
            assert isinstance(config["description"], str), f"{scene_type} description must be str"

    def test_aliases_are_lowercase(self):
        """All aliases should be lowercase for consistent matching."""
        for scene_type, config in SCENE_TYPES.items():
            for alias in config["aliases"]:
                assert alias == alias.lower(), f"Alias '{alias}' in {scene_type} should be lowercase"

    def test_canonical_names_follow_convention(self):
        """Canonical names should follow naming convention (interior_/exterior_/aerial_/twilight_/night_)."""
        valid_prefixes = ("interior_", "exterior_", "aerial_", "twilight_", "night_")
        for scene_type in SCENE_TYPES:
            assert any(
                scene_type.startswith(prefix) for prefix in valid_prefixes
            ), f"Scene type '{scene_type}' doesn't follow naming convention"


class TestNormalizeSceneType:
    """Tests for normalize_scene_type function."""

    # Interior scene types
    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("bedroom", "interior_bedroom"),
            ("bed", "interior_bedroom"),
            ("master", "interior_bedroom"),
            ("guest_room", "interior_bedroom"),
            ("suite", "interior_bedroom"),
            ("interior_bedroom", "interior_bedroom"),
        ],
    )
    def test_normalize_bedroom_aliases(self, raw_input, expected):
        """Test bedroom alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("great", "interior_great_room"),
            ("living", "interior_great_room"),
            ("family", "interior_great_room"),
            ("lounge", "interior_great_room"),
            ("great_room", "interior_great_room"),
            ("greatroom", "interior_great_room"),
        ],
    )
    def test_normalize_great_room_aliases(self, raw_input, expected):
        """Test great room alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("kitchen", "interior_kitchen"),
            ("kit", "interior_kitchen"),
            ("pantry", "interior_kitchen"),
            ("butler", "interior_kitchen"),
        ],
    )
    def test_normalize_kitchen_aliases(self, raw_input, expected):
        """Test kitchen alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("bathroom", "interior_bathroom"),
            ("bath", "interior_bathroom"),
            ("powder", "interior_bathroom"),
            ("powder_room", "interior_bathroom"),
        ],
    )
    def test_normalize_bathroom_aliases(self, raw_input, expected):
        """Test bathroom alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("dining", "interior_dining_room"),
            ("breakfast", "interior_dining_room"),
            ("dining_room", "interior_dining_room"),
        ],
    )
    def test_normalize_dining_aliases(self, raw_input, expected):
        """Test dining room alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("office", "interior_office"),
            ("study", "interior_office"),
            ("library", "interior_office"),
            ("den", "interior_office"),
        ],
    )
    def test_normalize_office_aliases(self, raw_input, expected):
        """Test office alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("closet", "interior_closet"),
            ("wardrobe", "interior_closet"),
            ("dressing", "interior_closet"),
            ("walk_in", "interior_closet"),
        ],
    )
    def test_normalize_closet_aliases(self, raw_input, expected):
        """Test closet alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("hallway", "interior_hallway"),
            ("corridor", "interior_hallway"),
            ("foyer", "interior_hallway"),
            # Note: "entry" overlaps with exterior_facade
        ],
    )
    def test_normalize_hallway_aliases(self, raw_input, expected):
        """Test hallway alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    # Exterior scene types
    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("pool", "exterior_pool"),
            ("spa", "exterior_pool"),
            ("water", "exterior_pool"),
            ("jacuzzi", "exterior_pool"),
            ("hot_tub", "exterior_pool"),
        ],
    )
    def test_normalize_pool_aliases(self, raw_input, expected):
        """Test pool alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("garden", "exterior_garden"),
            ("yard", "exterior_garden"),
            ("landscape", "exterior_garden"),
            ("landscaping", "exterior_garden"),
        ],
    )
    def test_normalize_garden_aliases(self, raw_input, expected):
        """Test garden alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("courtyard", "exterior_courtyard"),
            ("patio", "exterior_courtyard"),
            ("terrace", "exterior_courtyard"),
            ("deck", "exterior_courtyard"),
        ],
    )
    def test_normalize_courtyard_aliases(self, raw_input, expected):
        """Test courtyard alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("facade", "exterior_facade"),
            ("front", "exterior_facade"),
            ("exterior", "exterior_facade"),
            ("elevation", "exterior_facade"),
        ],
    )
    def test_normalize_facade_aliases(self, raw_input, expected):
        """Test facade alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("aerial", "aerial_exterior"),
            ("drone", "aerial_exterior"),
            ("overhead", "aerial_exterior"),
            ("bird", "aerial_exterior"),
            ("birds_eye", "aerial_exterior"),
        ],
    )
    def test_normalize_aerial_aliases(self, raw_input, expected):
        """Test aerial alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    # Special conditions
    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("twilight", "twilight_exterior"),
            ("dusk", "twilight_exterior"),
            ("blue_hour", "twilight_exterior"),
            ("golden_hour", "twilight_exterior"),
        ],
    )
    def test_normalize_twilight_aliases(self, raw_input, expected):
        """Test twilight alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("night", "night_interior"),
            ("evening_interior", "night_interior"),
            ("evening", "night_interior"),
        ],
    )
    def test_normalize_night_interior_aliases(self, raw_input, expected):
        """Test night interior alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    @pytest.mark.parametrize(
        "raw_input,expected",
        [
            ("night_exterior", "night_exterior"),
            ("nighttime_exterior", "night_exterior"),
        ],
    )
    def test_normalize_night_exterior_aliases(self, raw_input, expected):
        """Test night exterior alias normalization."""
        assert normalize_scene_type(raw_input) == expected

    # Case insensitivity
    def test_normalize_case_insensitive(self):
        """Normalization should be case-insensitive."""
        assert normalize_scene_type("MASTER") == normalize_scene_type("master")
        assert normalize_scene_type("Kitchen") == normalize_scene_type("kitchen")
        assert normalize_scene_type("POOL") == normalize_scene_type("pool")
        assert normalize_scene_type("Bedroom") == "interior_bedroom"

    # Whitespace handling
    def test_normalize_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert normalize_scene_type("  kitchen  ") == "interior_kitchen"
        assert normalize_scene_type("\tpool\n") == "exterior_pool"
        assert normalize_scene_type("   bedroom   ") == "interior_bedroom"

    # Canonical names (exact match)
    def test_normalize_canonical_names_exact_match(self):
        """Canonical names should return themselves."""
        for canonical_name in SCENE_TYPES:
            assert normalize_scene_type(canonical_name) == canonical_name

    # Error handling
    def test_normalize_invalid_raises_value_error(self):
        """Unknown types should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scene type"):
            normalize_scene_type("invalid_type")

    def test_normalize_empty_string_raises_value_error(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scene type"):
            normalize_scene_type("")

    def test_normalize_whitespace_only_raises_value_error(self):
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scene type"):
            normalize_scene_type("   ")

    def test_error_message_contains_valid_types(self):
        """Error message should list valid scene types."""
        try:
            normalize_scene_type("nonexistent_type")
        except ValueError as e:
            error_msg = str(e)
            # Should contain at least some canonical names
            assert "interior_bedroom" in error_msg or "interior_kitchen" in error_msg


class TestValidateSceneType:
    """Tests for validate_scene_type function."""

    def test_validate_all_canonical_types(self):
        """All canonical types should validate as true."""
        for scene_type in SCENE_TYPES:
            assert validate_scene_type(scene_type) is True

    def test_validate_invalid_type_returns_false(self):
        """Invalid types should return False, not raise."""
        assert validate_scene_type("invalid_type") is False
        assert validate_scene_type("") is False
        assert validate_scene_type("not_a_scene") is False

    def test_validate_alias_returns_false(self):
        """Aliases should return False (only canonical names valid)."""
        # "bedroom" is an alias, not the canonical name
        assert validate_scene_type("bedroom") is False
        assert validate_scene_type("pool") is False
        assert validate_scene_type("kitchen") is False


class TestGetSceneTypeDescription:
    """Tests for get_scene_type_description function."""

    def test_get_description_for_valid_types(self):
        """Should return description for valid canonical types."""
        for scene_type, config in SCENE_TYPES.items():
            result = get_scene_type_description(scene_type)
            assert result == config["description"]

    def test_get_description_returns_none_for_invalid(self):
        """Should return None for invalid types."""
        assert get_scene_type_description("invalid_type") is None
        assert get_scene_type_description("") is None

    def test_get_description_specific_values(self):
        """Test specific description values."""
        assert get_scene_type_description("interior_kitchen") == "Kitchens, butler's pantry"
        assert get_scene_type_description("exterior_pool") == "Pools, spas, water features"
        assert get_scene_type_description("aerial_exterior") == "Aerial views, drone shots"


class TestListSceneTypes:
    """Tests for list_scene_types function."""

    def test_list_returns_list(self):
        """Should return a list."""
        result = list_scene_types()
        assert isinstance(result, list)

    def test_list_not_empty(self):
        """Should return non-empty list."""
        result = list_scene_types()
        assert len(result) > 0

    def test_list_contains_all_canonical_types(self):
        """Should contain all canonical types from SCENE_TYPES."""
        result = list_scene_types()
        for scene_type in SCENE_TYPES:
            assert scene_type in result

    def test_list_matches_scene_types_keys(self):
        """List should match SCENE_TYPES keys exactly."""
        result = list_scene_types()
        assert set(result) == set(SCENE_TYPES.keys())

    def test_list_contains_expected_types(self):
        """Should contain expected scene types."""
        result = list_scene_types()
        expected = [
            "interior_bedroom",
            "interior_kitchen",
            "interior_bathroom",
            "exterior_pool",
            "aerial_exterior",
            "twilight_exterior",
        ]
        for expected_type in expected:
            assert expected_type in result


class TestGetAllAliases:
    """Tests for get_all_aliases function."""

    def test_get_aliases_returns_list(self):
        """Should return a list."""
        result = get_all_aliases("interior_bedroom")
        assert isinstance(result, list)

    def test_get_aliases_for_bedroom(self):
        """Test bedroom aliases."""
        aliases = get_all_aliases("interior_bedroom")
        expected = ["bedroom", "bed", "master", "guest_room", "suite"]
        assert set(aliases) == set(expected)

    def test_get_aliases_for_kitchen(self):
        """Test kitchen aliases."""
        aliases = get_all_aliases("interior_kitchen")
        expected = ["kitchen", "kit", "pantry", "butler"]
        assert set(aliases) == set(expected)

    def test_get_aliases_for_pool(self):
        """Test pool aliases."""
        aliases = get_all_aliases("exterior_pool")
        expected = ["pool", "spa", "water", "jacuzzi", "hot_tub"]
        assert set(aliases) == set(expected)

    def test_get_aliases_for_aerial(self):
        """Test aerial aliases."""
        aliases = get_all_aliases("aerial_exterior")
        expected = ["aerial", "drone", "overhead", "bird", "birds_eye"]
        assert set(aliases) == set(expected)

    def test_get_aliases_invalid_type_returns_empty(self):
        """Should return empty list for invalid types."""
        result = get_all_aliases("invalid_type")
        assert result == []

    def test_get_aliases_empty_string_returns_empty(self):
        """Should return empty list for empty string."""
        result = get_all_aliases("")
        assert result == []


class TestAliasUniqueness:
    """Tests for alias uniqueness across scene types."""

    def test_no_duplicate_aliases_within_type(self):
        """Aliases within a scene type should be unique."""
        for scene_type, config in SCENE_TYPES.items():
            aliases = config["aliases"]
            assert len(aliases) == len(set(aliases)), f"Duplicate aliases in {scene_type}"

    def test_no_unexpected_overlapping_aliases(self):
        """Verify no unexpected alias overlaps exist across scene types.

        Some aliases like 'entry' may match multiple scene types.
        This test documents known overlaps and fails if new ones are introduced.
        """
        # Collect all aliases and their mappings
        alias_to_types = {}
        for scene_type, config in SCENE_TYPES.items():
            for alias in config["aliases"]:
                if alias not in alias_to_types:
                    alias_to_types[alias] = []
                alias_to_types[alias].append(scene_type)

        # Known overlapping aliases that are intentionally shared
        known_overlaps = {"entry"}  # Used in both hallway and facade

        # Check for unexpected overlaps - fail if found
        unexpected_overlaps = []
        for alias, types in alias_to_types.items():
            if len(types) > 1 and alias not in known_overlaps:
                unexpected_overlaps.append((alias, types))

        assert not unexpected_overlaps, (
            f"Unexpected alias overlaps found: {unexpected_overlaps}. " "If intentional, add to known_overlaps set."
        )


class TestSceneTypeIntegration:
    """Integration tests for scene type module."""

    def test_roundtrip_normalization(self):
        """Normalized values should normalize to themselves."""
        for scene_type in list_scene_types():
            normalized = normalize_scene_type(scene_type)
            assert normalized == scene_type
            assert validate_scene_type(normalized)

    def test_alias_normalization_validates(self):
        """Normalized aliases should validate."""
        test_aliases = ["bedroom", "kitchen", "pool", "drone", "twilight"]
        for alias in test_aliases:
            normalized = normalize_scene_type(alias)
            assert validate_scene_type(normalized)

    def test_all_aliases_normalize_to_valid_types(self):
        """Every defined alias should normalize to a valid type.

        Note: Some aliases may be shared between scene types (e.g., 'entry').
        In these cases, the alias normalizes to whichever scene type is
        encountered first in dictionary iteration order.
        """
        # Known aliases that are intentionally shared
        known_overlapping_aliases = {"entry"}

        for scene_type, config in SCENE_TYPES.items():
            for alias in config["aliases"]:
                normalized = normalize_scene_type(alias)
                assert validate_scene_type(normalized), f"Alias '{alias}' normalized to invalid type"
                if alias not in known_overlapping_aliases:
                    assert (
                        normalized == scene_type
                    ), f"Alias '{alias}' should normalize to '{scene_type}' but got '{normalized}'"
