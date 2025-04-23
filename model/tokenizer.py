class SimpleTokenizer:
    def __init__(self):
        self.vocab = {chr(i+97): i/26 for i in range(26)}  # a-z to 0-1 range

    def tokenize(self, text):
        return [self.vocab.get(c.lower(), 0.5) for c in text if c.isalpha()][:10]

    def detokenize(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        result = ''
        for token in tokens:
            closest = min(self.vocab.values(), key=lambda x: abs(x-token))
            result += inv_vocab.get(closest, '?')
        return result
