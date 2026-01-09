import unittest
from unittest.mock import patch, MagicMock
from dreamsApp.app.utils.sentiment import get_chime_category

class TestChimeAnalysis(unittest.TestCase):

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
        result = get_chime_category(text)
        
        self.assertEqual(result['label'], 'Hope')
        self.assertEqual(result['score'], 0.95)
    
    @patch('dreamsApp.app.utils.sentiment.pipeline')
    def test_get_chime_category_empty(self, mock_pipeline):
        result = get_chime_category("")
        self.assertEqual(result['label'], 'Uncategorized')
        self.assertEqual(result['score'], 0.0)

    @patch('dreamsApp.app.utils.sentiment.pipeline')
    def test_get_chime_category_model_fail(self, mock_pipeline):
        # Simulate import error or download fail
        mock_pipeline.side_effect = Exception("Model not found")
        
        # We need to reset the global _chime_classifier to None for this test to trigger the exception block
        # However, it's global. We can patch the module level variable if needed, 
        # but since 'get_chime_category' interacts with it, we rely on it being None initially or reset it.
        import dreamsApp.app.utils.sentiment as sentiment_module
        sentiment_module._chime_classifier = None
        
        result = get_chime_category("some text")
        
        self.assertEqual(result['label'], 'Hope') # Fallback behavior
        self.assertEqual(result['score'], 0.0)
