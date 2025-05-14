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
# ────────────────────────────────────────────────────────────────────────
