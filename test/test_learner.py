import unittest
from model.learner import MemoryThief

class TestMemoryThief(unittest.TestCase):
    def setUp(self):
        self.ai = MemoryThief()

    def test_train_step(self):
        try:
            self.ai.train_step("This is a test input.")
        except Exception as e:
            self.fail(f"train_step raised an exception: {e}")

    def test_generate_text(self):
        result = self.ai.generate_text()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_save_and_load_model(self):
        self.ai.train_step("Memory test.")
        self.ai.save_model("memory/test_model.pth")

        new_ai = MemoryThief()
        try:
            new_ai.load_model("memory/test_model.pth")
        except Exception as e:
            self.fail(f"load_model raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
