from __future__ import annotations

from scratch_llm.tokenizer import ScratchTokenizer

EOS_TOKEN = "<|endoftext|>"


def resolve_eos_token_id(tokenizer: ScratchTokenizer) -> int | None:
    special_token_to_id = getattr(tokenizer, "special_token_to_id", None)
    if isinstance(special_token_to_id, dict) and EOS_TOKEN in special_token_to_id:
        return int(special_token_to_id[EOS_TOKEN])

    ids = tokenizer.encode(EOS_TOKEN, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None
