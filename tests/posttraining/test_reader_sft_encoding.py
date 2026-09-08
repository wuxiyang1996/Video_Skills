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


def test_completion_only_loss_uses_only_the_supervised_tail() -> None:
    import torch
    from trainer.reader.sft_lora import completion_only_loss

    V = 7
    calls = {}

    class _Out:
        def __init__(self, logits): self.logits = logits

    class _Model:
        def __call__(self, input_ids, attention_mask, logits_to_keep):
            calls["k"] = logits_to_keep
            T = input_ids.shape[1]
            # deterministic logits: position t strongly predicts token (t+1) % V
            logits = torch.full((1, logits_to_keep, V), -5.0)
            for j, t in enumerate(range(T - logits_to_keep, T)):
                logits[0, j, (t + 1) % V] = 5.0
            return _Out(logits)

    ids = torch.tensor([[(i) % V for i in range(12)]])
    labels = ids.clone(); labels[0, :9] = -100                       # 3 supervised tokens at the tail
    loss = completion_only_loss(_Model(), {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    assert calls["k"] == 4 and float(loss) < 0.01                    # only 4 positions materialised; predictions match labels
