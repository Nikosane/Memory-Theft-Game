import unittest
from model.tokenizer import SimpleTokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_tokenize_letters(self):
        result = self.tokenizer.tokenize("abc")
        self.assertTrue(all(0 <= x <= 1 for x in result))

    def test_tokenize_non_letters(self):
        result = self.tokenizer.tokenize("1234!?")
        self.assertEqual(result, [])

    def test_detokenize(self):
        tokens = self.tokenizer.tokenize("abc")
        text = self.tokenizer.detokenize(tokens)
        self.assertTrue(isinstance(text, str))

if __name__ == '__main__':
    unittest.main()
