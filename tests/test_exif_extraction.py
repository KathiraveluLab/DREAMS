"""Tests for EXIF extraction functionality."""

import pytest
from unittest.mock import patch, mock_open
from dreamsApp.exif_extractor import EXIFExtractor

class TestEXIFExtractor:
    
    def setup_method(self):
        self.extractor = EXIFExtractor()
    
    def test_empty_metadata_structure(self):
        """Test empty metadata has correct structure."""
        empty = self.extractor._empty_metadata()
        
        assert "timestamp" in empty
        assert "location" in empty
        assert "camera" in empty
        assert "processing" in empty
        assert empty["location"]["accuracy"] == "none"
    
    @patch('exifread.process_file')
    @patch('builtins.open', new_callable=mock_open)
    def test_exifread_extraction(self, mock_file, mock_process):
        """Test successful exifread extraction."""
        mock_process.return_value = {
            'EXIF DateTime': type('Tag', (), {'__str__': lambda: '2024:01:15 14:30:00'})(),
            'GPS GPSLatitude': type('Tag', (), {'__str__': lambda: '[61, 13, 4.68]'})(),
            'GPS GPSLatitudeRef': type('Tag', (), {'__str__': lambda: 'N'})(),
            'Image Make': type('Tag', (), {'__str__': lambda: 'Apple'})()
        }
        
        result = self.extractor._extract_exifread('test.jpg')
        
        assert result["processing"]["exif_source"] == "exifread"
        assert result["timestamp"] == "2024-01-15T14:30:00"
        assert result["camera"]["make"] == "Apple"
    
    def test_gps_coordinate_conversion(self):
        """Test GPS coordinate conversion to decimal."""
        tags = {
            'GPS GPSLatitude': type('Tag', (), {'__str__': lambda: '[61, 13, 4.68]'})(),
            'GPS GPSLatitudeRef': type('Tag', (), {'__str__': lambda: 'N'})()
        }
        
        coord = self.extractor._get_gps_coordinate(tags, 'GPS GPSLatitude', 'GPS GPSLatitudeRef')
        
        assert coord is not None
        assert abs(coord - 61.2180) < 0.001  # Approximate check
    
    def test_missing_gps_data(self):
        """Test handling of missing GPS data."""
        tags = {}
        
        coord = self.extractor._get_gps_coordinate(tags, 'GPS GPSLatitude', 'GPS GPSLatitudeRef')
        
        assert coord is None
    
    @patch('dreamsApp.exif_extractor.EXIFExtractor._extract_exifread')
    @patch('dreamsApp.exif_extractor.EXIFExtractor._extract_pillow')
    def test_fallback_strategy(self, mock_pillow, mock_exifread):
        """Test fallback from exifread to Pillow."""
        mock_exifread.side_effect = Exception("exifread failed")
        mock_pillow.return_value = {"processing": {"exif_source": "pillow"}}
        
        result = self.extractor.extract_metadata('test.jpg')
        
        assert result["processing"]["exif_source"] == "pillow"
        mock_exifread.assert_called_once()
        mock_pillow.assert_called_once()