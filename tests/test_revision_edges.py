import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from alignment.batching import build_sft_batch, encode_row
from alignment.io import digest
from alignment.losses import compute_sft_loss, response_logps
from alignment.training import encode_records
from alignment.formatting import format_prompt
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_empty_prompt_explicitly_leaves_first_response_token_unscoreable():
    tokenizer = ToyTokenizer(); model = ToyCausalLM(tokenizer.vocab_size)
    batch = build_sft_batch(tokenizer, [""], ["ab"], 8)
    score = response_logps(model, batch)
    expected = -2 * torch.log(torch.tensor(float(tokenizer.vocab_size)))
    torch.testing.assert_close(score[0], expected)


def test_training_counts_and_warns_for_empty_tokenized_formatted_prompt(monkeypatch):
    tokenizer = ToyTokenizer()
    original = tokenizer.encode
    empty_formatted = format_prompt("q")
    monkeypatch.setattr(tokenizer, "encode", lambda text, add_special_tokens=False:
                        [] if text == empty_formatted else original(text, add_special_tokens))
    with pytest.warns(RuntimeWarning, match="1 records have an empty tokenized formatted prompt"):
        rows = encode_records(tokenizer, [{"id": "edge", "prompt": "q", "response": "ab"}],
                              {"track": "sft", "max_length": 16})
    assert len(rows) == 1


def test_empty_prompt_warning_is_emitted_once_for_multiple_rows(monkeypatch):
    tokenizer = ToyTokenizer(); original = tokenizer.encode
    formatted = {format_prompt("q1"), format_prompt("q2")}
    monkeypatch.setattr(tokenizer, "encode", lambda text, add_special_tokens=False:
                        [] if text in formatted else original(text, add_special_tokens))
    with pytest.warns(RuntimeWarning) as caught:
        rows = encode_records(tokenizer, [
            {"id":"a","prompt":"q1","response":"ab"},
            {"id":"b","prompt":"q2","response":"cd"}],
            {"track":"sft","max_length":16})
    assert len(rows) == 2 and len(caught) == 1
    assert "2 records" in str(caught[0].message)


def test_full_prompt_zero_token_row_has_finite_scalar_loss():
    tokenizer = ToyTokenizer(); model = ToyCausalLM(tokenizer.vocab_size)
    loss = compute_sft_loss(model, tokenizer, ["abcd"], ["xy"], 4)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() == 0


def test_eos_removed_by_truncation_and_response_entirely_truncated():
    tokenizer = ToyTokenizer()
    row = encode_row(tokenizer, "ab", "cd", 4)
    assert row["input_ids"][-1] != tokenizer.eos_token_id
    truncated = encode_row(tokenizer, "abcd", "xy", 4)
    assert truncated["labels"] == [-100] * 4


def test_mixed_lengths_keep_row_order():
    tokenizer = ToyTokenizer()
    batch = build_sft_batch(tokenizer, ["a", "long"], ["x", "yz"], 12)
    assert batch["input_ids"][0, 0].item() == tokenizer.vocab["a"]
    assert batch["input_ids"][1, 0].item() == tokenizer.vocab["l"]
    assert batch["attention_mask"].sum(1).tolist() == [3, 7]


def test_tiny_optimization_step_lowers_response_loss():
    tokenizer = ToyTokenizer(); model = ToyCausalLM(tokenizer.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = compute_sft_loss(model, tokenizer, ["p"], ["x"], 8)
    optimizer.zero_grad(); before.backward(); optimizer.step()
    after = compute_sft_loss(model, tokenizer, ["p"], ["x"], 8)
    assert after < before


def test_judge_cli_accepts_manual_underscore_flags_and_prints_summary(tmp_path, monkeypatch, capsys):
    import sys
    import alignment.judging as judging
    from alignment.cli import main
    pair = {"id":"p1","prompt_id":"r1","track":"hh","instruction":"x","A":"a","B":"b",
            "mapping":{"A":"base","B":"candidate"},"candidate":"candidate","order_seed":42}
    pair["hash"] = digest(pair)
    pairs = tmp_path/"pairs.jsonl"; pairs.write_text(json.dumps(pair)+"\n",encoding="utf-8")
    output = tmp_path/"judged.jsonl"
    monkeypatch.setattr(judging, "kaggle_backend", lambda model: lambda prompt: [{"id":"p1","decision":"B"}])
    monkeypatch.setattr(sys, "argv", ["judge", "--pairs_path", str(pairs), "--output_path", str(output),
        "--judge_model", "google/gemini-3.6-flash", "--rubric", "hh", "--batch_size", "5",
        "--sleep_seconds", "5", "--max_retries", "3", "--reuse_judgments"])
    main("judge_pairwise_kaggle")
    assert "W=1 L=0 T=0 U=0 F=0" in capsys.readouterr().out


class FakeBatch(dict):
    def __getattr__(self, name): return self[name]
    def to(self, device): return self


class FakeTokenizer:
    pad_token_id=0; eos_token_id=1; chat_template="fake"
    def apply_chat_template(self, rows, tokenize=False, add_generation_prompt=True): return "CHAT:"+rows[0]["content"]
    def __call__(self, text, **kwargs): return FakeBatch(input_ids=torch.tensor([[2,3]]),attention_mask=torch.tensor([[1,1]]))
    def decode(self, ids, skip_special_tokens=False): return "answer<eos>"


class FakeGenerator:
    def generate(self, **kwargs): return torch.tensor([[2,3,4,1]])


def test_generation_sanity_record_contains_required_evidence():
    from scripts.sanity_check_generation import inspect_rows
    rows=inspect_rows({"model":"m","device":"cpu","format":"chat","max_new_tokens":3},
                      FakeTokenizer(),FakeGenerator(),[{"id":"p","instruction":"hello"}])
    assert rows[0]["formatted_prompt"] == "CHAT:hello"
    assert rows[0]["eos_present"] is True and rows[0]["token_count"] == 2


def test_decoded_length_proxy_is_labelled_as_retokenized_not_cap_hit():
    from scripts.analyze_results import output_stats

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(int(text)))

    stats = output_stats([{"output": "255"}, {"output": "256"}], Tokenizer())
    assert stats["retokenized_decoded_length_ge_256_pct"] == 50.0
    assert "hit_256_token_cap_pct" not in stats
