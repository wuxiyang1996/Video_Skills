from trainer.reader.sft_lora import encode_example


class _Tok:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "|".join(m["content"] for m in messages) + "|A:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_prompt_is_masked_and_completion_supervised_with_eos() -> None:
    row = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}], "completion": "{ok}"}
    enc = encode_example(_Tok(), row, max_len=64)
    n_prompt = len("sys|usr|A:")
    assert enc["labels"][:n_prompt] == [-100] * n_prompt
    assert enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:] and enc["input_ids"][-1] == 0
    assert encode_example(_Tok(), row, max_len=12) is None       # prompt too long -> dropped
