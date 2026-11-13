"""Tests for location enrichment functions (reverse geocoding + semantic embedding).

All tests use mocks — no real API calls or model loading.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
from PIL.ExifTags import IFD, Base
from PIL.TiffImagePlugin import IFDRational

import dreams_app.core.location_extractor as location_extractor

# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_NOMINATIM_RESPONSE = {
    "display_name": "St. Mary's Church, Main St, Anchorage, AK, USA",
    "type": "place_of_worship",
    "category": "amenity",
    "address": {
        "amenity": "St. Mary's Church",
        "road": "Main Street",
        "city": "Anchorage",
        "state": "Alaska",
        "country": "United States",
        "country_code": "us",
    },
}

PARK_NOMINATIM_RESPONSE = {
    "display_name": "Westchester Lagoon, Spenard Rd, Anchorage, AK, USA",
    "type": "park",
    "category": "leisure",
    "address": {
        "leisure": "Westchester Lagoon",
        "road": "Spenard Rd",
        "city": "Anchorage",
        "state": "Alaska",
        "country": "United States",
        "country_code": "us",
    },
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache and rate-limit state between tests."""
    location_extractor._geocode_cache.clear()
    location_extractor._last_request_time = 0.0
    yield


# ── TestReverseGeocode ────────────────────────────────────────────────────


class TestReverseGeocode:
    """5 tests for reverse_geocode()."""

    @patch("dreams_app.core.location_extractor.requests.get")
    @patch("dreams_app.core.location_extractor.time.sleep")
    def test_successful_geocode(self, mock_sleep, mock_get):
        """Successful API call returns expected dict structure."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_NOMINATIM_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = location_extractor.reverse_geocode(61.2181, -149.9003)

        assert result is not None
        assert result["display_name"] == "St. Mary's Church, Main St, Anchorage, AK, USA"
        assert result["place_category"] == "place_of_worship"
        assert result["place_type"] == "amenity"

    @patch("dreams_app.core.location_extractor.requests.get")
    @patch("dreams_app.core.location_extractor.time.sleep")
    def test_cache_deduplication(self, mock_sleep, mock_get):
        """Second call with same coords should hit cache, not API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_NOMINATIM_RESPONSE
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result1 = location_extractor.reverse_geocode(61.2181, -149.9003)
        result2 = location_extractor.reverse_geocode(61.2181, -149.9003)

        assert result1 == result2
        assert mock_get.call_count == 1  # only one HTTP call

    def test_invalid_coordinates_returns_none(self):
        """Out-of-range lat/lon should return None without calling API."""
        assert location_extractor.reverse_geocode(91.0, 0.0) is None
        assert location_extractor.reverse_geocode(0.0, 181.0) is None
        assert location_extractor.reverse_geocode(-91.0, 0.0) is None

    @patch("dreams_app.core.location_extractor.requests.get")
    @patch("dreams_app.core.location_extractor.time.sleep")
    def test_network_error_returns_none(self, mock_sleep, mock_get):
        """Network failure should return None gracefully."""
        import requests as req_lib
        mock_get.side_effect = req_lib.ConnectionError("connection refused")

        result = location_extractor.reverse_geocode(61.2181, -149.9003)
        assert result is None

    @patch("dreams_app.core.location_extractor.requests.get")
    @patch("dreams_app.core.location_extractor.time.sleep")
    def test_nominatim_error_returns_none(self, mock_sleep, mock_get):
        """Nominatim error response (e.g. 'Unable to geocode') returns None."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "Unable to geocode"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = location_extractor.reverse_geocode(61.2181, -149.9003)
        assert result is None


# ── TestFormatLocationText ────────────────────────────────────────────────


class TestFormatLocationText:
    """4 tests for format_location_text()."""

    def test_full_result_has_place_type_not_geography(self):
        """Output contains place category + type + name, NOT city/state."""
        geocode = {
            "display_name": "St. Mary's Church, Main St, Anchorage, AK, USA",
            "place_category": "place_of_worship",
            "place_type": "amenity",
            "address": {
                "amenity": "St. Mary's Church",
                "road": "Main Street",
                "city": "Anchorage",
                "state": "Alaska",
            },
        }
        text = location_extractor.format_location_text(geocode, 61.2181, -149.9003)

        assert "place of worship" in text
        assert "amenity" in text
        assert "St. Mary's Church" in text
        # Geography should NOT appear
        assert "Anchorage" not in text
        assert "Alaska" not in text

    def test_none_fallback(self):
        """None geocode result falls back to coordinate string."""
        text = location_extractor.format_location_text(None, 61.2181, -149.9003)
        assert text == "unknown location at coordinates 61.2181 -149.9003"

    def test_park_with_place_name(self):
        """Park uses leisure class key to find the place name."""
        geocode = {
            "display_name": "Westchester Lagoon, Spenard Rd, Anchorage, AK, USA",
            "place_category": "park",
            "place_type": "leisure",
            "address": {
                "leisure": "Westchester Lagoon",
                "road": "Spenard Rd",
            },
        }
        text = location_extractor.format_location_text(geocode, 61.2, -149.9)
        assert "park" in text
        assert "leisure" in text
        assert "Westchester Lagoon" in text

    def test_no_duplication_when_category_equals_type(self):
        """When category and type are the same, don't repeat the word."""
        geocode = {
            "display_name": "Some Road, Anchorage, AK, USA",
            "place_category": "residential",
            "place_type": "residential",
            "address": {},
        }
        text = location_extractor.format_location_text(geocode, 61.2, -149.9)
        # "residential" should appear exactly once
        assert text.count("residential") == 1


# ── TestGetLocationEmbedding ──────────────────────────────────────────────


class TestGetLocationEmbedding:
    """2 tests for get_location_embedding()."""

    def test_returns_384_dim_list(self):
        """Embedding should be a list of 384 floats."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

        embedding = location_extractor.get_location_embedding(
            "place of worship amenity St. Mary's Church",
            model=mock_model,
        )

        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_passes_text_to_model(self):
        """The model.encode() should receive the exact text string."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        text = "hospital amenity Alaska Native Medical Center"
        location_extractor.get_location_embedding(text, model=mock_model)

        mock_model.encode.assert_called_once_with(text)


# ── TestEnrichLocation ────────────────────────────────────────────────────


class TestEnrichLocation:
    """2 tests for enrich_location()."""

    @patch("dreams_app.core.location_extractor.reverse_geocode")
    def test_success_with_geocode(self, mock_geocode):
        """Successful geocode populates all fields."""
        mock_geocode.return_value = {
            "display_name": "St. Mary's Church, Main St, Anchorage, AK, USA",
            "place_category": "place_of_worship",
            "place_type": "amenity",
            "address": {
                "amenity": "St. Mary's Church",
                "city": "Anchorage",
                "state": "Alaska",
                "country": "United States",
                "country_code": "us",
            },
        }

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(384, dtype=np.float32)

        result = location_extractor.enrich_location(61.2181, -149.9003, model=mock_model)

        assert "location_text" in result
        assert "location_embedding" in result
        assert "display_name" in result
        assert "place_category" in result
        assert "place_type" in result
        assert "address" in result
        assert len(result["location_embedding"]) == 384

    @patch("dreams_app.core.location_extractor.reverse_geocode")
    def test_fallback_when_geocode_fails(self, mock_geocode):
        """When geocoding fails, still returns text + embedding."""
        mock_geocode.return_value = None

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        result = location_extractor.enrich_location(61.2181, -149.9003, model=mock_model)

        assert "location_text" in result
        assert "unknown location" in result["location_text"]
        assert "location_embedding" in result
        assert len(result["location_embedding"]) == 384
        # Should NOT have geocode-specific fields
        assert "display_name" not in result
        assert "place_category" not in result


# ── TestExtractGps ────────────────────────────────────────────────────────


def _write_photo(path, lat_ref="N", lon_ref="W", gps_time=True, gps=True,
                 lat_degrees=61, lon_degrees=149):
    """Save a small JPEG with EXIF laid out the way cameras write it."""
    exif = Image.Exif()
    if gps:
        gps_ifd = {
            1: lat_ref,
            2: (IFDRational(lat_degrees), IFDRational(13), IFDRational(513, 100)),
            3: lon_ref,
            4: (IFDRational(lon_degrees), IFDRational(54), IFDRational(108, 100)),
        }
        if gps_time:
            gps_ifd[7] = (IFDRational(10), IFDRational(30), IFDRational(15))
            gps_ifd[29] = "2024:06:01"
        exif[IFD.GPSInfo] = gps_ifd
    exif[IFD.Exif] = {Base.DateTimeOriginal: "2024:06:01 02:30:15"}
    Image.new("RGB", (8, 8), "white").save(str(path), exif=exif)
    return str(path)


class TestExtractGps:
    """Tests for extract_gps_from_image()."""

    def test_reads_coordinates(self, tmp_path):
        """Geotagged photo returns signed decimal degrees."""
        result = location_extractor.extract_gps_from_image(
            _write_photo(tmp_path / "a.jpg")
        )

        assert result["lat"] == pytest.approx(61.218092, abs=1e-5)
        assert result["lon"] == pytest.approx(-149.9003, abs=1e-5)
        assert result["timestamp"] == "2024-06-01T10:30:15+00:00"

    def test_southern_and_eastern_refs(self, tmp_path):
        """S latitude is negative, E longitude stays positive."""
        result = location_extractor.extract_gps_from_image(
            _write_photo(tmp_path / "a.jpg", lat_ref="S", lon_ref="E")
        )

        assert result["lat"] == pytest.approx(-61.218092, abs=1e-5)
        assert result["lon"] == pytest.approx(149.9003, abs=1e-5)

    def test_falls_back_to_datetime_original(self, tmp_path):
        """Without a GPS timestamp the capture time is used."""
        result = location_extractor.extract_gps_from_image(
            _write_photo(tmp_path / "a.jpg", gps_time=False)
        )

        assert result["timestamp"] == "2024-06-01T02:30:15"

    @pytest.mark.parametrize("kwargs", [
        {"lat_degrees": 95},
        {"lon_degrees": 200},
    ])
    def test_out_of_range_coordinates_return_none(self, tmp_path, kwargs):
        """Corrupt EXIF that decodes to impossible coordinates is not returned."""
        result = location_extractor.extract_gps_from_image(
            _write_photo(tmp_path / "a.jpg", **kwargs)
        )

        assert result is None

    def test_nan_coordinates_return_none(self, tmp_path, monkeypatch):
        """A NaN decoded from corrupt EXIF is rejected, not returned."""
        monkeypatch.setattr(location_extractor, "_dms_to_decimal", lambda dms: float("nan"))

        result = location_extractor.extract_gps_from_image(_write_photo(tmp_path / "a.jpg"))

        assert result is None

    def test_no_gps_returns_none(self, tmp_path):
        """Photo with EXIF but no GPS block returns None."""
        result = location_extractor.extract_gps_from_image(
            _write_photo(tmp_path / "a.jpg", gps=False)
        )

        assert result is None
