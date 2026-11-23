import unittest
from unittest.mock import patch, MagicMock
from dreams_app.core.chime_classifier import chime_score_map
from dreams_app.core.sentiment import SentimentAnalyzer

class TestChimeAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a fresh analyzer for each test to avoid cross-test contamination
        self.analyzer = SentimentAnalyzer()

    @patch('dreams_app.core.sentiment.pipeline')
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
    
    @patch('dreams_app.core.sentiment.pipeline')
    def test_chime_result_keeps_the_full_score_distribution(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=[[
            {'label': 'Hope', 'score': 0.6},
            {'label': 'Connectedness', 'score': 0.3},
            {'label': 'Identity', 'score': 0.05},
            {'label': 'Meaning', 'score': 0.03},
            {'label': 'Empowerment', 'score': 0.02},
        ]])

        result = self.analyzer.analyze_chime("I feel hopeful and close to my family.")

        self.assertEqual(result['label'], 'Hope')
        self.assertEqual(result['score'], 0.6)
        self.assertEqual(result['scores'], {
            'Hope': 0.6, 'Connectedness': 0.3, 'Identity': 0.05,
            'Meaning': 0.03, 'Empowerment': 0.02,
        })

    @patch('dreams_app.core.sentiment.pipeline')
    def test_get_chime_category_empty(self, mock_pipeline):
        result = self.analyzer.analyze_chime("")
        self.assertEqual(result['label'], 'Uncategorized')
        self.assertEqual(result['score'], 0.0)

    @patch('dreams_app.core.sentiment.pipeline')
    def test_get_chime_category_model_fail(self, mock_pipeline):
        # Simulate import error or download fail
        mock_pipeline.side_effect = Exception("Model not found")
        
        result = self.analyzer.analyze_chime("some text")
        
        self.assertEqual(result['label'], 'Uncategorized')
        self.assertEqual(result['score'], 0.0)



class TestChimeScoreMap(unittest.TestCase):

    def test_flat_and_nested_outputs_give_the_same_map(self):
        flat = [{'label': 'Hope', 'score': 0.7}, {'label': 'Meaning', 'score': 0.3}]

        self.assertEqual(chime_score_map(flat), {'Hope': 0.7, 'Meaning': 0.3})
        self.assertEqual(chime_score_map([flat]), {'Hope': 0.7, 'Meaning': 0.3})

    def test_unknown_labels_and_bad_items_are_ignored(self):
        results = [
            {'label': 'Hope', 'score': 0.5},
            {'label': 'LABEL_7', 'score': 0.4},
            {'label': 'Identity', 'score': 'not a number'},
            {'label': 'Meaning'},
            'junk',
        ]

        self.assertEqual(chime_score_map(results), {'Hope': 0.5})

    def test_nothing_usable_gives_an_empty_map(self):
        self.assertEqual(chime_score_map([]), {})
        self.assertEqual(chime_score_map(None), {})
        self.assertEqual(chime_score_map([[]]), {})
