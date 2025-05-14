import regex as re
from helper import highlight_tokens

class BasicTokenizer():
    def __init__(self):
        self.merges: dict[tuple[int,int],int] = {}

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
        for pair, idx in self.merges.items():
            ids = self.merge(ids, pair, idx)
        return ids


    def decode(self, ids):
        # build vocab with raw utf values
        vocab = { idx: bytes([idx]) for idx in range(256)} # maps from integer representation to raw byte
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

class RegexTokenizer(BasicTokenizer):
    def __init__(self):
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.pattern = re.compile(self.GPT4_SPLIT_PATTERN)

    def get_stats(self, ids_list):
        counts = {}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text, vocab_size, verbose=False):
        num_merges = max(0, vocab_size - 256)  # UTF-8 has 256 values
        matched = re.findall(self.pattern, text)
        ids_list = [list(word.encode("utf-8")) for word in matched] # convert text to list of integer representations of UTF-8 bytes
        self.merges = {}
        for i in range(num_merges):
            stats = self.get_stats(ids_list)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            for i, ids in enumerate(ids_list):
                ids_list[i] = self.merge(ids, pair, idx)
                self.merges[pair] = idx
        return [token for ids in ids_list for token in ids]
    
    def encode(self, text):
        matched = re.findall(self.pattern, text)
        ids_list = []
        for match in matched:
            ids_list.append(super().encode(match))
        return [token for ids in ids_list for token in ids]
    
    def decode(self, ids):
        return super().decode(ids)


tok1 = BasicTokenizer()
tok2 = RegexTokenizer()

msg = "Train your tokenizer on whatever text you like and visualize the merged tokens. Do they look reasonable? One default test you may wish to use is the text file tests/taylorswift.txt."
output1 = tok1.train(msg, 280)
output2 = tok2.train(msg, 280)

encoded1 = tok1.encode(msg)
encoded2 = tok2.encode(msg)
print(tok1.decode(encoded1) == tok2.decode(encoded2))

highlight_tokens(encoded1, tok1)
highlight_tokens(encoded2, tok2)