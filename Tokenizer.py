import regex as re
import tiktoken
from helper import recover_merges

class BasicTokenizer():
    def __init__(self, _merges: dict[tuple[int,int],int]={}, byte_shuffle={}, special_tokens=[]):
        self._merges = _merges
        self.byte_shuffle = byte_shuffle
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.special_tokens = {tok: 256 + len(self._merges) + i for i, tok in enumerate(special_tokens)}
        self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        """(Re)compute self.vocab from _merges + byte shuffle once."""
        vocab = {i: bytes([i]) for i in range(256)}          # raw bytes
        for (p0, p1), idx in self._merges.items():            # _merged pairs
            vocab[idx] = vocab[p0] + vocab[p1]
        # add special tokens as UTF-8
        for tok, idx in self.special_tokens.items():
            vocab[idx] = tok.encode("utf-8")
        self.vocab = vocab

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx): # Replace every occurence of {pair} in {ids} with {idx}.
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
        num__merges = max(0, vocab_size - 256)  # UTF-8 has 256 values
        ids = list(text.encode("utf-8")) # convert text to list of integer representations of UTF-8 bytes
        self._merges = {}
        for i in range(num__merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self._merges[pair] = idx
        self._rebuild_vocab()
        return ids
            

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            for tok, tok_id in self.special_tokens.items(): # check for special tokens
                if text.startswith(tok, i):
                    ids.append(tok_id)
                    i += len(tok)
                    break
            else:
                b = text[i].encode("utf-8")
                if self.byte_shuffle:
                    ids.append(self.byte_shuffle.get(b, b))
                else:
                    ids.extend(b)
                i += 1

        for pair, idx in self._merges.items():
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        if self.inverse_byte_shuffle:
            text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        return text_bytes.decode("utf-8", errors="replace")
    
class RegexTokenizer(BasicTokenizer):
    def __init__(self, _merges: dict[tuple[int,int],int]={}, byte_shuffle={}, special_tokens=[]):
        super().__init__(_merges, byte_shuffle, special_tokens)
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.pattern = re.compile(self.GPT4_SPLIT_PATTERN)

    def _get_stats(self, ids_list):
        counts = {}
        for ids in ids_list:
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text, vocab_size, verbose=False):
        num__merges = max(0, vocab_size - 256)  # UTF-8 has 256 values
        matched = re.findall(self.pattern, text)
        ids_list = [list(word.encode("utf-8")) for word in matched] # convert text to list of integer representations of UTF-8 bytes
        self._merges = {}
        for i in range(num__merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            for j, ids in enumerate(ids_list):
                ids_list[j] = self._merge(ids, pair, idx)
            self._merges[pair] = idx
        self._rebuild_vocab()
        return [token for ids in ids_list for token in ids]
    
    def encode(self, text):
        matched = re.findall(self.pattern, text)
        ids_list = []
        for match in matched:
            ids_list.append(super().encode(match))
        return [token for ids in ids_list for token in ids]

with open("tests/taylorswift.txt") as f:
    training = f.read()

msg = "<|endoftext|>hello world"

# Initialize RegexTokenizer with the special token
regex_tokenizer = RegexTokenizer(
    special_tokens=["<|endoftext|>"]
)

regex_tokenizer.train(training, 380)

# Encode with RegexTokenizer
regex_ids = regex_tokenizer.encode(msg)
print("RegexTokenizer IDs:    ", regex_ids)

# Use tiktoken's GPT-4 tokenizer for comparison
tiktok = tiktoken.get_encoding("cl100k_base")
tiktok_ids = tiktok.encode(msg, allowed_special="all")
print("tiktoken (cl100k_base) IDs:", tiktok_ids)

# Compare the two ID sequences
print("Match:", regex_ids == tiktok_ids)

# Decode both back to text
print("RegexTokenizer decode:", regex_tokenizer.decode(regex_ids))
print("tiktoken decode:      ", tiktok.decode(tiktok_ids))