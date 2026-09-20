"""
Tests for the constraint engine and the PLAN-ACT-OBSERVE-REFLECT loop.

A FakeModel stands in for the SLM so the control flow can be tested
deterministically, with no checkpoint and no GPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import (  # noqa: E402
    StoryDirectorAgent,
    contains_word,
    excludes_word,
    min_words,
)


class FakeEnc:
    """Minimal stand-in for a tiktoken encoding."""

    def encode_ordinary(self, text):
        return [len(w) for w in text.split()] or [0]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class FakeModel:
    """Returns a scripted sequence of outputs, one per generate() call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return [out]


class ScriptedAgent(StoryDirectorAgent):
    """StoryDirectorAgent whose _generate returns the model's scripted string."""

    def _generate(self, prompt, max_new_tokens, temperature):
        return self.model.generate(None, max_new_tokens, temperature)[0]


# --- constraints ----------------------------------------------------------- #

def test_contains_word_is_case_insensitive_by_default():
    c = contains_word("sandwich")
    assert c("He ate a SANDWICH.")
    assert not c("He ate a pear.")


def test_contains_word_case_sensitive():
    c = contains_word("Sandwich", case_sensitive=True)
    assert c("A Sandwich appeared.")
    assert not c("a sandwich appeared.")


def test_min_words():
    assert min_words(3)("one two three")
    assert not min_words(4)("one two three")


def test_excludes_word():
    assert excludes_word("dragon")("a quiet story")
    assert not excludes_word("dragon")("a DRAGON appeared")


# --- the loop -------------------------------------------------------------- #

def test_succeeds_on_first_attempt_without_retrying():
    model = FakeModel(["the dog ate a sandwich"])
    agent = ScriptedAgent(model, FakeEnc(), verbose=False)
    res = agent.generate_directed_story("the dog", "sandwich")
    assert res.success
    assert res.n_attempts == 1
    assert model.calls == 1


def test_retries_then_succeeds_after_reflection():
    model = FakeModel(["the dog ran", "the dog found a sandwich"])
    agent = ScriptedAgent(model, FakeEnc(), verbose=False)
    res = agent.generate_directed_story("the dog", "sandwich")
    assert res.success
    assert res.n_attempts == 2
    # The second attempt must have run against an injected strategy prompt.
    assert res.attempts[0].strategy is not None
    assert "sandwich" in res.attempts[0].strategy


def test_gives_up_after_max_retries_and_reports_failure():
    model = FakeModel(["no target word here"])
    agent = ScriptedAgent(model, FakeEnc(), verbose=False)
    res = agent.generate_directed_story("the dog", "sandwich", max_retries=3)
    assert not res.success
    assert res.n_attempts == 3
    assert model.calls == 3


def test_all_constraints_must_pass_together():
    model = FakeModel(["a sandwich"])           # has the word, too short
    agent = ScriptedAgent(model, FakeEnc(), verbose=False)
    res = agent.run(
        "the dog",
        constraints=[contains_word("sandwich"), min_words(10)],
        max_retries=2,
    )
    assert not res.success
    assert "min_words:10" in res.attempts[-1].failed
    assert "contains:sandwich" in res.attempts[-1].passed


def test_trace_is_recorded_for_every_attempt():
    model = FakeModel(["miss", "miss", "hit sandwich"])
    agent = ScriptedAgent(model, FakeEnc(), verbose=False)
    res = agent.generate_directed_story("the dog", "sandwich")
    assert res.success
    assert [a.index for a in res.attempts] == [1, 2, 3]
    assert res.attempts[0].failed == ["contains:sandwich"]
    assert res.attempts[-1].ok


def test_build_strategies_all_mention_the_required_word():
    strategies = StoryDirectorAgent.build_strategies("Once upon a time", "balloon")
    assert len(strategies) == 5
    assert all("balloon" in s for s in strategies)
    assert all(s.startswith("Once upon a time") for s in strategies)
