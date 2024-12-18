import unittest

from agent_service.tools.web_search.brightdata_websearch import parse_news_result, parse_web_result


class TestWebSearch(unittest.TestCase):
    def test_parse_news_result(self):
        test = {
            "news": [
                {"title": "title 1", "link": "link 1"},
                {"title": "title 2", "link": "link 2"},
                {"title": "title 3", "link": "link 3"},
                {"title": "title 4", "link": "link 4"},
            ]
        }
        val = parse_news_result(test, 3)
        expected_val = ["link 1", "link 2", "link 3"]
        self.assertEqual(val, expected_val)

    def test_parse_web_result(self):
        test = {
            "organic": [
                {"title": "title 1", "link": "link 1"},
                {"title": "title 2", "link": "link 2"},
                {"title": "title 3", "link": "link 3"},
                {"title": "title 4", "link": "link 4"},
            ]
        }
        val = parse_web_result(test, 3)
        expected_val = ["link 1", "link 2", "link 3"]
        self.assertEqual(val, expected_val)
