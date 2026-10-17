import copy
import math
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
import submission_adapter as a
from alignment.losses import dpo_metrics
from tests.toy_lm import ToyTokenizer, ToyCausalLM


@pytest.mark.parametrize("cap", [1, 3, 4, 5, 6, 9])
def test_eos_appended_before_truncation(cap):
    tok = ToyTokenizer()
    batch = a.build_sft_batch(tok, ["abc"], ["de"], cap)
    expected = tok.encode("abcde") + [tok.eos_token_id]
    assert batch["input_ids"][0].tolist() == expected[:cap]
    assert batch["labels"][0].tolist() == ([-100] * 3 + tok.encode("de") + [1])[:cap]


@pytest.mark.parametrize("pad", [0, 1, None])
def test_padding_is_positional_even_when_pad_is_eos(pad):
    tok = ToyTokenizer()
    tok.pad_token_id = pad
    batch = a.build_sft_batch(tok, ["a", "abc"], ["b", "def"], 20)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0, 0, 0]
    assert batch["labels"][0].tolist() == [-100, tok.vocab["b"], 1, -100, -100, -100, -100]
    assert tok.pad_token_id == pad


def test_absent_special_tokens_and_empty_input():
    tok = ToyTokenizer()
    tok.eos_token_id = tok.pad_token_id = None
    batch = a.build_sft_batch(tok, ["", "a"], ["", "b"], 8)
    assert batch["labels"][0].eq(-100).all()
    assert batch["attention_mask"][0].eq(0).all()
    assert batch["labels"][1].tolist() == [-100, tok.vocab["b"]]
    model = ToyCausalLM(tok.vocab_size)
    assert a.compute_response_logprobs(model, tok, [""], [""], 8).item() == 0
    loss = a.compute_sft_loss(model, tok, [""], [""], 8)
    assert loss.item() == 0 and loss.ndim == 0
    loss.backward()
    assert model.logit_bias.grad.eq(0).all()


def test_existing_eos_is_preserved_plus_contract_append():
    tok = ToyTokenizer()
    batch = a.build_sft_batch(tok, ["a"], ["b<eos>"], 8)
    assert batch["input_ids"][0].tolist() == tok.encode("ab") + [1, 1]


class TransitionLM(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(vocab * vocab, dtype=torch.float64).reshape(vocab, vocab).sin())

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.table[input_ids])


def test_causal_shift_first_response_eos_and_row_order_hand_computed():
    tok = ToyTokenizer()
    model = TransitionLM(tok.vocab_size)
    prompts, responses = ["xy", "a", "abc"], ["ab", "c", ""]
    actual = a.compute_response_logprobs(model, tok, prompts, responses, 20)
    expected = []
    for prompt, response in zip(prompts, responses):
        ids = tok.encode(prompt + response) + [1]
        expected.append(sum((model.table[ids[i - 1], ids[i]] - torch.logsumexp(model.table[ids[i - 1]], 0))
                            for i in range(len(prompt), len(ids))))
    torch.testing.assert_close(actual, torch.stack(expected))
    loss = a.compute_sft_loss(model, tok, prompts, responses, 20)
    torch.testing.assert_close(loss, -torch.stack(expected).sum() / 6)
    loss.backward()
    assert model.table.grad.abs().sum() > 0
    # No gradient through prompt-only predecessor x -> y.
    assert model.table.grad[tok.vocab['x']].eq(0).all()


def test_truncated_prompt_and_unconditioned_first_token_have_zero_score():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    logps = a.compute_response_logprobs(model, tok, ["abcdef", ""], ["z", "a"], 1)
    torch.testing.assert_close(logps, torch.zeros(2))


def test_batch_invariance_to_neighbors_and_order():
    tok = ToyTokenizer()
    model = TransitionLM(tok.vocab_size)
    p, r = ["a", "longprompt", "xyz"], ["q", "longresponse", "r"]
    whole = a.compute_response_logprobs(model, tok, p, r, 40)
    for i in range(3):
        torch.testing.assert_close(whole[i:i+1], a.compute_response_logprobs(model, tok, p[i:i+1], r[i:i+1], 40))
    permutation = [2, 0, 1]
    torch.testing.assert_close(whole[permutation], a.compute_response_logprobs(model, tok, [p[i] for i in permutation], [r[i] for i in permutation], 40))


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.7, 10.0])
def test_dpo_matches_independent_softplus_and_gradients(beta):
    pc = torch.tensor([-1., -9., -3.], dtype=torch.float64, requires_grad=True)
    pr = torch.tensor([-5., -2., -3.], dtype=torch.float64, requires_grad=True)
    rc = torch.tensor([-2., -4., -6.], dtype=torch.float64, requires_grad=True)
    rr = torch.tensor([-3., -7., -6.], dtype=torch.float64, requires_grad=True)
    z = beta * ((pc - pr) - (rc.detach() - rr.detach()))
    expected = torch.logaddexp(torch.zeros_like(z), -z).mean()
    loss = a.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(pc.grad, -beta * torch.sigmoid(-z.detach()) / 3)
    torch.testing.assert_close(pr.grad, -pc.grad)
    assert rc.grad is None and rr.grad is None


def test_extreme_dpo_is_finite_and_scalar():
    pc = torch.tensor([-1e6, 1e6], requires_grad=True)
    zero = torch.zeros(2)
    loss = a.compute_dpo_loss_from_logps(pc, zero, zero, zero, 1.)
    assert loss.ndim == 0 and loss.item() == 500000
    loss.backward()
    assert torch.isfinite(pc.grad).all()


def test_reference_no_grad_eval_frozen_and_policy_optimizer_step():
    tok = ToyTokenizer()
    class Spy(ToyCausalLM):
        def forward(self, *args, **kwargs):
            assert not torch.is_grad_enabled()
            assert not self.training
            return super().forward(*args, **kwargs)
    policy, ref = ToyCausalLM(tok.vocab_size), Spy(tok.vocab_size)
    original = copy.deepcopy(ref.state_dict())
    before = policy.logit_bias.detach().clone()
    opt = torch.optim.SGD(policy.parameters(), lr=.5)
    loss = a.compute_dpo_loss(policy, ref, tok, ["q"], ["a"], ["b"], .2, 20)
    assert abs(loss.item() - math.log(2)) < 1e-6
    loss.backward()
    assert policy.logit_bias.grad.abs().sum() > 0
    opt.step()
    assert not torch.equal(before, policy.logit_bias)
    assert all(not p.requires_grad and p.grad is None for p in ref.parameters())
    for k, v in ref.state_dict().items():
        torch.testing.assert_close(v, original[k])


def test_dpo_side_order_and_strict_tie_metric():
    tok = ToyTokenizer()
    b = a.build_dpo_batch(tok, ["a", "b"], ["cc", "d"], ["e", "fff"], 10)
    for side, responses in [("chosen", ["cc", "d"]), ("rejected", ["e", "fff"])]:
        expected = a.build_sft_batch(tok, ["a", "b"], responses, 10)
        for key in expected:
            assert torch.equal(expected[key], b[side][key])
    z = torch.zeros(2)
    assert dpo_metrics(z, z, z, z, .1) == dict(chosen_reward=0, rejected_reward=0, reward_accuracy=0, reward_margin=0)


def test_invalid_inputs_fail_instead_of_silent_zip_truncation():
    tok = ToyTokenizer()
    with pytest.raises(ValueError): a.build_sft_batch(tok, ["a"], [], 8)
    with pytest.raises(ValueError): a.build_sft_batch(tok, [], [], 8)
    with pytest.raises(ValueError): a.build_dpo_batch(tok, ["a"], ["b"], [], 8)
    with pytest.raises(ValueError): a.build_sft_batch(tok, ["a"], ["b"], 0)
    with pytest.raises(ValueError): a.compute_dpo_loss_from_logps(torch.zeros(2,1), torch.zeros(2), torch.zeros(2), torch.zeros(2), .1)
    model = ToyCausalLM(tok.vocab_size)
    with pytest.raises(ValueError): a.compute_dpo_loss(model, model, tok, ["a"], ["b"], ["c"], .1, 8)
