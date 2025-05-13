

class BasicTokenizer():
    def __init__(self):
        self.merges = {}

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx): # Replace every occurence of {pair} in {ids} with {idx}.
        new_ids = []
        i = 0
        while i < len(ids): # build up the new list new_ids by appending either the replaced token or the original tokens
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
            

    def train(self, text, vocab_size, verbose=False):
        num_merges = max(0, vocab_size - 256)  # UTF-8 has 256 values
        ids = list(text.encode("utf-8")) # convert text to list of integer representations of UTF-8 bytes
        # TODO: do the regex splitting that OpenAI does
        self.merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        return ids
            

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, idx in self.merges:
            ids = self.merge(ids, pair, idx)
        return ids


    def decode(self, ids):
        pass

tok = BasicTokenizer()

msg = "Train brain ainainain test ainainain"
output = tok.train(msg, 260)
print(msg)
print(output)
print(tok.merges)