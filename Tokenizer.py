import regex as re
import tiktoken
from helper import highlight_tokens, recover_merges

class BasicTokenizer():
    def __init__(self, merges: dict[tuple[int,int],int]={}, byte_shuffle={}):
        self.merges = merges
        self.byte_shuffle = byte_shuffle
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

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
        raw_bytes = text.encode("utf-8")
        if self.byte_shuffle:
            ids = [ self.byte_shuffle.get(b, b) for b in raw_bytes ]
        else:
            ids = list(raw_bytes)
        for pair, idx in self.merges.items():
            ids = self.merge(ids, pair, idx)
        return ids


    def decode(self, ids):
        # build vocab with raw utf values
        vocab = { idx: bytes([idx]) for idx in range(256)} # maps from integer representation to raw byte
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        tokens = b"".join(vocab[idx] for idx in ids)
        if self.inverse_byte_shuffle:
            tokens = bytes(self.inverse_byte_shuffle[b] for b in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text

class RegexTokenizer(BasicTokenizer):
    def __init__(self, merges: dict[tuple[int,int],int]={}, byte_shuffle={}):
        super().__init__(merges, byte_shuffle)
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
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            for j, ids in enumerate(ids_list):
                ids_list[j] = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        return [token for ids in ids_list for token in ids]
    
    def encode(self, text):
        matched = re.findall(self.pattern, text)
        ids_list = []
        for match in matched:
            ids_list.append(super().encode(match))
        return [token for ids in ids_list for token in ids]


msg = "hello world!!!? (안녕하세요!) lol123 😉"

enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids1 = enc.encode(msg)
text1 = enc.decode(ids1) # get the same text back

byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
merges = recover_merges(enc._mergeable_ranks)
tok = RegexTokenizer(merges, byte_shuffle)
ids2 = tok.encode(msg)
text2 = tok.decode(ids2)

print(ids1)
print(ids2)
print(text1)
print(text2)

new_tok = RegexTokenizer()
new_tok.train("""
Unfortunately, you will run into two issues:

It is not trivial to recover the raw merges from the GPT-4 tokenizer. You can easily recover what we call vocab here, and what they call and store under enc._mergeable_ranks. Feel free to copy paste the recover_merges function in minbpe/gpt4.py, which takes these ranks and returns the raw merges. If you wish to know how this function works, read this and this. Basically, under some conditions it is enough to only store the parent nodes (and their rank) and get rid of the precise details of which children merged up to any parent.
Second, the GPT-4 tokenizer for some reason permutes its raw bytes. It stores this permutation in the first 256 elements of the mergeable ranks, so you can recover this byte shuffle relatively simply as byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}. In both your encode and decode, you'll have to shuffle bytes around accordingly. If you're stuck, reference the minbpe/gpt4.py` file for hints.
""", 275)
ids3 = new_tok.encode(msg)
text3 = new_tok.decode(ids3)
print(ids3)
print(text3)
