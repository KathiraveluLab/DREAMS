"""
Unit tests for analysis_pipeline.utils

Covers:
- make_memory_id  — determinism, uniqueness
- hamming_distance — identical / different / mismatched lengths
- perceptual_hash  — identical images, distinct images
- clean_text       — @-mention and URL replacement
- parse_timestamp  — ISO-8601 variants, bad input
- safe_float / safe_int — happy paths + edge cases
- validate_safe_path    — allowed & forbidden paths
- is_image_file         — extension allow-list
"""

import pytest
from pathlib import Path
from PIL import Image

from analysis_pipeline.utils import (
    make_memory_id,
    hamming_distance,
    perceptual_hash,
    clean_text,
    parse_timestamp,
    safe_float,
    safe_int,
    validate_safe_path,
    is_image_file,
)


# ── make_memory_id ────────────────────────────────────────────────────────────

class TestMakeMemoryId:
    def test_deterministic(self):
        assert make_memory_id("alice", "photo.jpg") == make_memory_id("alice", "photo.jpg")

    def test_different_users_differ(self):
        assert make_memory_id("alice", "photo.jpg") != make_memory_id("bob", "photo.jpg")

    def test_different_filenames_differ(self):
        assert make_memory_id("alice", "a.jpg") != make_memory_id("alice", "b.jpg")

    def test_returns_16_char_hex(self):
        mid = make_memory_id("u1", "img.png")
        assert len(mid) == 16
        int(mid, 16)  # raises if not valid hex


# ── hamming_distance ──────────────────────────────────────────────────────────

class TestHammingDistance:
    def test_identical_hashes_give_zero(self):
        assert hamming_distance("ff00ff00", "ff00ff00") == 0

    def test_opposite_hashes_give_nonzero(self):
        assert hamming_distance("ff00ff00", "00ff00ff") > 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            hamming_distance("abcd", "abcdef")

    def test_single_bit_difference(self):
        # Only the last hex digit differs (f vs e → 1111 vs 1110, 1 bit)
        dist = hamming_distance("fffffffe", "ffffffff")
        assert dist == 1


# ── perceptual_hash ───────────────────────────────────────────────────────────

class TestPerceptualHash:
    def _solid(self, color):
        return Image.new("RGB", (64, 64), color=color)

    def test_identical_images_same_hash(self):
        img = self._solid((100, 100, 100))
        assert perceptual_hash(img) == perceptual_hash(img)

    def test_different_images_different_hash(self):
        h1 = perceptual_hash(self._solid((0, 0, 0)))
        h2 = perceptual_hash(self._solid((255, 255, 255)))
        assert h1 != h2

    def test_hash_is_hex_string(self):
        h = perceptual_hash(self._solid((128, 64, 32)))
        int(h, 16)  # raises if not valid hex

    def test_hash_length_matches_hash_size(self):
        from analysis_pipeline.config import IMAGE_HASH_SIZE
        h = perceptual_hash(self._solid((50, 50, 50)))
        expected_len = IMAGE_HASH_SIZE * IMAGE_HASH_SIZE // 4
        assert len(h) == expected_len


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_replaces_mention(self):
        assert clean_text("Hello @John") == "Hello @user"

    def test_replaces_url(self):
        assert clean_text("Visit https://example.com now") == "Visit http now"

    def test_replaces_http_url(self):
        assert clean_text("See http://foo.bar/baz") == "See http"

    def test_multiple_mentions(self):
        result = clean_text("@alice and @bob talked")
        assert "@alice" not in result
        assert "@bob" not in result
        assert "@user" in result

    def test_plain_text_unchanged(self):
        text = "A beautiful sunny day in the park."
        assert clean_text(text) == text

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"


# ── parse_timestamp ───────────────────────────────────────────────────────────

class TestParseTimestamp:
    def test_none_returns_none(self):
        assert parse_timestamp(None) is None

    def test_empty_string_returns_none(self):
        assert parse_timestamp("") is None

    def test_whitespace_returns_none(self):
        assert parse_timestamp("   ") is None

    def test_iso_with_timezone(self):
        dt = parse_timestamp("2026-03-01T10:30:00+05:30")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3

    def test_iso_with_microseconds(self):
        dt = parse_timestamp("2026-03-01T11:06:00.327879+00:00")
        assert dt is not None
        assert dt.microsecond == 327879

    def test_date_only(self):
        dt = parse_timestamp("2025-12-25")
        assert dt is not None
        assert dt.day == 25

    def test_us_format(self):
        dt = parse_timestamp("03/15/2025 08:00:00")
        assert dt is not None
        assert dt.month == 3

    def test_unparseable_returns_none(self):
        assert parse_timestamp("not-a-date") is None


# ── safe_float / safe_int ─────────────────────────────────────────────────────

class TestSafeFloat:
    def test_valid_string(self):
        assert safe_float("3.14") == pytest.approx(3.14)

    def test_valid_number(self):
        assert safe_float(42) == 42.0

    def test_none_returns_default(self):
        assert safe_float(None) is None
        assert safe_float(None, default=0.0) == 0.0

    def test_nan_returns_default(self):
        assert safe_float("nan") is None

    def test_inf_returns_default(self):
        assert safe_float("inf") is None

    def test_bad_string_returns_default(self):
        assert safe_float("abc", default=-1.0) == -1.0


class TestSafeInt:
    def test_valid(self):
        assert safe_int("7") == 7

    def test_none_returns_none(self):
        assert safe_int(None) is None

    def test_float_string_fails(self):
        # int("3.5") raises ValueError
        assert safe_int("3.5") is None


# ── validate_safe_path ────────────────────────────────────────────────────────

class TestValidateSafePath:
    def test_allowed_root(self, tmp_path):
        p = tmp_path / "sub" / "file.txt"
        result = validate_safe_path(p, allowed_roots=[tmp_path])
        assert result == p.resolve()

    def test_outside_root_raises(self, tmp_path):
        other = tmp_path / ".." / ".."
        with pytest.raises(ValueError):
            validate_safe_path(other, allowed_roots=[tmp_path / "safe"])

    def test_no_allowed_roots_returns_resolved(self, tmp_path):
        p = tmp_path / "file.txt"
        result = validate_safe_path(p)
        assert result == p.resolve()


# ── is_image_file ─────────────────────────────────────────────────────────────

class TestIsImageFile:
    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"])
    def test_allowed_extensions(self, ext):
        assert is_image_file(f"photo{ext}") is True

    @pytest.mark.parametrize("ext", [".txt", ".csv", ".pdf", ".mp4", ""])
    def test_disallowed_extensions(self, ext):
        assert is_image_file(f"file{ext}") is False

    def test_uppercase_extension(self):
        # extensions should be matched case-insensitively
        assert is_image_file("PHOTO.JPG") is True
