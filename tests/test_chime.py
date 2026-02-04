import unittest
from unittest.mock import patch, MagicMock
from dreamsApp.app.utils.sentiment import SentimentAnalyzer

class TestChimeAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a fresh analyzer for each test to avoid cross-test contamination
        self.analyzer = SentimentAnalyzer()

    @patch('dreamsApp.app.utils.sentiment.pipeline')
    def test_get_chime_category_success(self, mock_pipeline):
        # Mock the pipeline return value
        mock_classifier = MagicMock()
        mock_classifier.return_value = [[
            {'label': 'Hope', 'score': 0.95},
            {'label': 'Connectedness', 'score': 0.02},
            {'label': 'Identity', 'score': 0.01},
            {'label': 'Meaning', 'score': 0.01},
            {'label': 'Empowerment', 'score': 0.01}
        ]]
        mock_pipeline.return_value = mock_classifier
        
        text = "I feel hopeful about the future."
        result = self.analyzer.analyze_chime(text)
        
        self.assertEqual(result['label'], 'Hope')
        self.assertEqual(result['score'], 0.95)
    
    @patch('dreamsApp.app.utils.sentiment.pipeline')
    def test_get_chime_category_empty(self, mock_pipeline):
        result = self.analyzer.analyze_chime("")
        self.assertEqual(result['label'], 'Uncategorized')
        self.assertEqual(result['score'], 0.0)

    @patch('dreamsApp.app.utils.sentiment.pipeline')
    def test_get_chime_category_model_fail(self, mock_pipeline):
        # Simulate import error or download fail
        mock_pipeline.side_effect = Exception("Model not found")
        
        result = self.analyzer.analyze_chime("some text")
        
        self.assertEqual(result['label'], 'Uncategorized')
        self.assertEqual(result['score'], 0.0)

