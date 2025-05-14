# helper.py ─────────────────────────────────────────────────────────────
RESET = "\x1b[0m"

def _ansi_bg(code: int) -> str:
    """Return an ANSI-256 *background*-colour escape sequence."""
    return f"\x1b[48;5;{code}m"      # 48 ⇒ background, 5 ⇒ 256-colour table

def highlight_tokens(ids, tokenizer):
    """
    Print the decoded text with a unique background colour for every
    distinct token.  Re-uses the same background whenever the same
    token id appears again (like most web visualisers).
    """
    id2bg   = {}          # token-id → escape sequence
    next_bg = 16          # start of the 6×6×6 colour cube (codes 16-231)

    out = []
    for idx in ids:
        # lazily assign a background colour to new token ids
        if idx not in id2bg:
            id2bg[idx] = _ansi_bg(next_bg)
            next_bg    = 16 + ((next_bg - 15) % 216) + 1   # cycle through cube

        out.append(f"{id2bg[idx]}{tokenizer.decode([idx])}{RESET}")

    print("".join(out))

def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges