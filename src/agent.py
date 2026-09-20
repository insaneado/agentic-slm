"""
StoryDirectorAgent - a Plan / Act / Observe / Reflect control loop around the SLM.

A 15M-parameter model follows instructions unreliably: asked for a story that
must contain a specific word, it frequently ignores the constraint. Rather than
scaling the model, this agent wraps it in a validation loop.

    PLAN     build a generation plan from the goal (prompt + required word)
    ACT      sample a draft from the SLM
    OBSERVE  run deterministic constraint checks over the draft
    REFLECT  on failure, pick a different prompt-injection strategy and retry

The observer is deterministic (string matching and logic checks), so the loop
adds reliability without adding a second model.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #

@dataclass
class Constraint:
    """A named, deterministic check over generated text."""
    name: str
    check: Callable[[str], bool]
    describe: str = ""

    def __call__(self, text: str) -> bool:
        return bool(self.check(text))


def contains_word(word: str, case_sensitive: bool = False) -> Constraint:
    """The generated text must contain `word`."""
    def _check(text: str) -> bool:
        return (word in text) if case_sensitive else (word.lower() in text.lower())
    return Constraint(
        name=f"contains:{word}",
        check=_check,
        describe=f"output must contain the word '{word}'",
    )


def min_words(n: int) -> Constraint:
    """The generated text must be at least `n` words long."""
    return Constraint(
        name=f"min_words:{n}",
        check=lambda text: len(text.split()) >= n,
        describe=f"output must be at least {n} words",
    )


def excludes_word(word: str) -> Constraint:
    """The generated text must NOT contain `word`."""
    return Constraint(
        name=f"excludes:{word}",
        check=lambda text: word.lower() not in text.lower(),
        describe=f"output must not contain the word '{word}'",
    )


# --------------------------------------------------------------------------- #
# Trace records - every attempt is recorded so runs are auditable
# --------------------------------------------------------------------------- #

@dataclass
class Attempt:
    index: int
    prompt: str
    text: str
    passed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    strategy: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass
class DirectorResult:
    success: bool
    text: str
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class StoryDirectorAgent:
    """Drives a small language model until its output satisfies every constraint.

    Args:
        model:   a GPT instance exposing `.generate(idx, max_new_tokens, temperature)`
        enc:     a tiktoken encoding (GPT-2 BPE)
        device:  torch device string
        verbose: print the PLAN/ACT/OBSERVE/REFLECT trace as it runs
    """

    def __init__(self, model, enc, device="cpu", verbose=True):
        self.model = model
        self.enc = enc
        self.device = device
        self.verbose = verbose

    # -- internal helpers --------------------------------------------------- #

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """ACT: sample one draft from the SLM."""
        input_ids = self.enc.encode_ordinary(prompt)
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.enc.decode(out[0].tolist())

    @staticmethod
    def _observe(text: str, constraints: List[Constraint]):
        """OBSERVE: run every constraint, return (passed, failed) names."""
        passed, failed = [], []
        for c in constraints:
            (passed if c(text) else failed).append(c.name)
        return passed, failed

    @staticmethod
    def build_strategies(start_prompt: str, required_word: str) -> List[str]:
        """REFLECT: prompt-injection tactics, tried in order after a failure.

        Each one seeds the required word into the context so the model is far
        more likely to carry it into the continuation.
        """
        return [
            f"{start_prompt} Suddenly, a huge {required_word} appeared right in front of them.",
            f"{start_prompt} They looked down and saw a {required_word} on the ground.",
            f"{start_prompt} Look! It is a {required_word}! they shouted.",
            f"{start_prompt} The most important thing in the room was the {required_word}.",
            f"{start_prompt} Everyone was talking about the magical {required_word}.",
        ]

    # -- public API --------------------------------------------------------- #

    def run(
        self,
        start_prompt: str,
        constraints: List[Constraint],
        strategies: Optional[List[str]] = None,
        max_retries: int = 4,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
    ) -> DirectorResult:
        """Run the full PLAN-ACT-OBSERVE-REFLECT loop.

        Returns a DirectorResult carrying the final text and the full attempt
        trace, whether or not it succeeded.
        """
        # PLAN
        self._log("[PLAN]  goal: " + start_prompt)
        for c in constraints:
            self._log(f"        constraint: {c.describe or c.name}")

        current_prompt = start_prompt
        attempts: List[Attempt] = []

        for i in range(1, max_retries + 1):
            # ACT
            text = self._generate(current_prompt, max_new_tokens, temperature)

            # OBSERVE
            passed, failed = self._observe(text, constraints)
            attempt = Attempt(
                index=i, prompt=current_prompt, text=text,
                passed=passed, failed=failed,
            )
            preview = text.replace("\n", " ")[:100]
            self._log(f"[ACT]    attempt {i}: {preview}...")

            if not failed:
                self._log(f"[OBSERVE] all {len(passed)} constraint(s) satisfied")
                attempts.append(attempt)
                return DirectorResult(success=True, text=text, attempts=attempts)

            self._log(f"[OBSERVE] failed: {', '.join(failed)}")

            # REFLECT - swap in a new strategy for the next attempt
            if i < max_retries:
                if strategies is None:
                    required = failed[0].split(":", 1)[-1]
                    strategies = self.build_strategies(start_prompt, required)
                new_prompt = strategies[i % len(strategies)]
                attempt.strategy = new_prompt
                self._log(f"[REFLECT] switching tactic -> '...{new_prompt[-40:]}'")
                current_prompt = new_prompt

            attempts.append(attempt)

        self._log(
            f"[DONE]   exhausted {max_retries} attempts without satisfying all constraints"
        )
        return DirectorResult(success=False, text=attempts[-1].text, attempts=attempts)

    def generate_directed_story(
        self, start_prompt: str, required_word: str, max_retries: int = 4
    ) -> DirectorResult:
        """Convenience wrapper: 'write a story that contains `required_word`'."""
        return self.run(
            start_prompt=start_prompt,
            constraints=[contains_word(required_word)],
            strategies=self.build_strategies(start_prompt, required_word),
            max_retries=max_retries,
        )
