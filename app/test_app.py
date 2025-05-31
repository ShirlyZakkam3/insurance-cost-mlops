import unittest
import json
from main import app  

class InsuranceAPITestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Insurance Cost Predictor', response.data)

    def test_prediction_success(self):
        payload = {
            "age": 35,
            "sex": "male",
            "bmi": 28.5,
            "children": 2,
            "smoker": "no",
            "region": "southwest"
        }
        response = self.app.post('/predict', data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_cost', response.get_json())

    def test_prediction_missing_field(self):
        payload = {
            "age": 35,
            "sex": "male",
            # Missing 'bmi'
            "children": 2,
            "smoker": "no",
            "region": "southwest"
        }
        response = self.app.post('/predict', data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', response.get_json())

if __name__ == '__main__':
    unittest.main()
