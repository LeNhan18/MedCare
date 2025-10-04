import unittest
from src.api.app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'healthy'})

    def test_symptom_to_medication(self):
        response = self.app.post('/api/recommend', json={'symptoms': 'headache'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('medications', response.json)

    def test_invalid_symptom(self):
        response = self.app.post('/api/recommend', json={'symptoms': ''})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.json)

if __name__ == '__main__':
    unittest.main()