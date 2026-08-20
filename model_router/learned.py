from __future__ import annotations

import json
import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np

MODEL_SCHEMA_VERSION = "complexity-router-nb-v2"
SUPPORTED_MODEL_SCHEMAS = frozenset({"complexity-router-nb-v1", MODEL_SCHEMA_VERSION})
VECTORIZER_VERSION = "hashed-word-ngram-v1"
DEFAULT_FEATURE_DIMENSION = 32_768
WORD_PATTERN = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+", re.IGNORECASE)
Whitespace = re.compile(r"\s+")
DecisionPolicy = Literal["argmax", "expected", "conservative", "tier_risk", "adaptive"]


def level_to_tier(level: int) -> int:
    return {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}[level]


def tier_underroute_probability(probabilities: np.ndarray, selected_tier: int) -> float:
    if selected_tier not in {1, 2, 3}:
        raise ValueError("selected_tier must be 1, 2, or 3")
    if selected_tier == 1:
        return float(probabilities[2:].sum())
    if selected_tier == 2:
        return float(probabilities[3:].sum())
    return 0.0


def tier_for_risk(probabilities: np.ndarray, tolerance: float) -> int:
    """Return the cheapest tier whose posterior under-route risk is tolerated."""
    if not 0 <= tolerance < 1:
        raise ValueError("underroute_tolerance must be in [0, 1)")
    if tier_underroute_probability(probabilities, 1) <= tolerance:
        return 1
    if tier_underroute_probability(probabilities, 2) <= tolerance:
        return 2
    return 3


def normalize_prompt(text: str) -> str:
    return Whitespace.sub(" ", text.strip().lower())


def default_artifact_path() -> Path:
    return Path(
        str(files("model_router").joinpath("artifacts/complexity_router_v3.npz"))
    )


def final_user_turn(text: str) -> str:
    matches = list(re.finditer(r"(?:^|\n)\s*user:\s*", text, re.IGNORECASE))
    return text[matches[-1].end() :].strip() if matches else text


@dataclass(frozen=True)
class ComplexityPrediction:
    level: int
    expected_level: float
    confidence: float
    entropy: float
    probabilities: tuple[float, ...]
    decision_policy: str
    minimum_tier: int
    tier_underroute_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "expected_level": round(self.expected_level, 4),
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "probabilities": {
                str(index + 1): round(probability, 5)
                for index, probability in enumerate(self.probabilities)
            },
            "decision_policy": self.decision_policy,
            "minimum_tier": self.minimum_tier,
            "tier_underroute_probability": round(self.tier_underroute_probability, 5),
        }


class HashedTextVectorizer:
    """Dependency-light, stable word/unigram feature extraction."""

    def __init__(
        self,
        dimension: int = DEFAULT_FEATURE_DIMENSION,
        max_tokens: int = 4_096,
    ) -> None:
        if dimension < 1_024:
            raise ValueError("feature dimension must be at least 1024")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        self.dimension = dimension
        self.max_tokens = max_tokens

    def _feature_counts(self, text: str) -> Counter[str]:
        raw_tokens = WORD_PATTERN.findall(text.lower())
        tokens = ["<num>" if token.isdigit() else token for token in raw_tokens]
        if len(tokens) > self.max_tokens:
            # The final turns are normally most important for multi-turn routing.
            tokens = tokens[-self.max_tokens :]

        counts: Counter[str] = Counter(f"word={token}" for token in tokens)
        counts.update(
            f"bigram={first}|{second}"
            for first, second in zip(tokens, tokens[1:], strict=False)
        )

        final_user_turn = text.rsplit("User:", 1)[-1]
        counts.update(
            [
                f"__chars={min(15, len(text) // 100)}",
                f"__words={min(20, len(tokens) // 20)}",
                f"__user_turns={min(8, text.count('User:'))}",
                f"__assistant_turns={min(8, text.count('Assistant:'))}",
                f"__final_chars={min(15, len(final_user_turn) // 50)}",
                f"__questions={min(5, text.count('?'))}",
                f"__code_fences={min(4, text.count('```'))}",
            ]
        )
        return counts

    def transform_one(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        counts = self._feature_counts(text)
        indices = np.fromiter(
            (
                zlib.crc32(feature.encode("utf-8")) % self.dimension
                for feature in counts
            ),
            dtype=np.int32,
            count=len(counts),
        )
        values = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        return indices, values


class NaiveBayesComplexityModel:
    """Five-level prompt complexity classifier trained from audited JSONL."""

    def __init__(
        self,
        log_feature_probability: np.ndarray,
        log_class_prior: np.ndarray,
        *,
        temperature: float = 1.0,
        final_turn_weight: float = 0.0,
        vectorizer_version: str = VECTORIZER_VERSION,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if log_feature_probability.ndim != 2:
            raise ValueError("log_feature_probability must be a 2D array")
        if log_feature_probability.shape[0] != 5:
            raise ValueError("the complexity model must contain five classes")
        if log_class_prior.shape != (5,):
            raise ValueError("log_class_prior must contain five values")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 <= final_turn_weight <= 1:
            raise ValueError("final_turn_weight must be in [0, 1]")
        if vectorizer_version != VECTORIZER_VERSION:
            raise ValueError(
                f"Unsupported vectorizer {vectorizer_version!r}; "
                f"expected {VECTORIZER_VERSION!r}"
            )
        self.log_feature_probability = np.asarray(
            log_feature_probability, dtype=np.float64
        )
        self.log_class_prior = np.asarray(log_class_prior, dtype=np.float64)
        self.temperature = float(temperature)
        self.final_turn_weight = float(final_turn_weight)
        self.vectorizer_version = vectorizer_version
        self.metadata = metadata or {}
        self.vectorizer = HashedTextVectorizer(
            dimension=self.log_feature_probability.shape[1]
        )

    @property
    def feature_dimension(self) -> int:
        return self.log_feature_probability.shape[1]

    def raw_scores(self, prompt: str) -> np.ndarray:
        indices, values = self.vectorizer.transform_one(prompt)
        feature_scores = (self.log_feature_probability[:, indices] * values).sum(axis=1)
        return self.log_class_prior + feature_scores

    def probabilities(self, prompt: str) -> np.ndarray:
        probabilities = self._probabilities_for_view(prompt)
        latest = final_user_turn(prompt)
        if self.final_turn_weight and latest != prompt:
            latest_probabilities = self._probabilities_for_view(latest)
            probabilities = (
                1.0 - self.final_turn_weight
            ) * probabilities + self.final_turn_weight * latest_probabilities
        return probabilities / probabilities.sum()

    def _probabilities_for_view(self, prompt: str) -> np.ndarray:
        scores = self.raw_scores(prompt) / self.temperature
        scores -= scores.max()
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum()

    def predict(
        self,
        prompt: str,
        *,
        decision_policy: DecisionPolicy = "argmax",
        underroute_tolerance: float = 0.20,
    ) -> ComplexityPrediction:
        probabilities = self.probabilities(prompt)
        expected_level = float(np.dot(probabilities, np.arange(1, 6, dtype=np.float64)))

        if decision_policy == "argmax":
            level = int(np.argmax(probabilities)) + 1
        elif decision_policy == "expected":
            level = max(1, min(5, int(math.floor(expected_level + 0.5))))
        elif decision_policy == "conservative":
            if not 0 <= underroute_tolerance < 1:
                raise ValueError("underroute_tolerance must be in [0, 1)")
            level = (
                int(
                    np.searchsorted(
                        np.cumsum(probabilities), 1.0 - underroute_tolerance
                    )
                )
                + 1
            )
            level = min(5, level)
        elif decision_policy == "tier_risk":
            minimum_tier = tier_for_risk(probabilities, underroute_tolerance)
            # Preserve ordinal meaning while making the selected tier explicit.
            level = {1: 2, 2: 3, 3: 5}[minimum_tier]
        elif decision_policy == "adaptive":
            raise ValueError(
                "adaptive policy requires request context; use the router classifier"
            )
        else:
            raise ValueError(f"Unknown decision policy: {decision_policy}")

        minimum_tier = (
            minimum_tier if decision_policy == "tier_risk" else level_to_tier(level)
        )
        underroute_probability = tier_underroute_probability(
            probabilities, minimum_tier
        )

        entropy = -float(
            np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        ) / math.log(5)
        return ComplexityPrediction(
            level=level,
            expected_level=expected_level,
            confidence=float(probabilities.max()),
            entropy=entropy,
            probabilities=tuple(float(value) for value in probabilities),
            decision_policy=decision_policy,
            minimum_tier=minimum_tier,
            tier_underroute_probability=underroute_probability,
        )

    def save(self, path: str | Path) -> None:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            schema_version=np.asarray([MODEL_SCHEMA_VERSION]),
            vectorizer_version=np.asarray([self.vectorizer_version]),
            log_feature_probability=self.log_feature_probability,
            log_class_prior=self.log_class_prior,
            temperature=np.asarray([self.temperature], dtype=np.float64),
            final_turn_weight=np.asarray([self.final_turn_weight], dtype=np.float64),
            metadata_json=np.asarray(
                [json.dumps(self.metadata, sort_keys=True, ensure_ascii=False)]
            ),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> NaiveBayesComplexityModel:
        artifact_path = Path(path) if path else default_artifact_path()
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Learned router artifact not found: {artifact_path}"
            )
        with np.load(artifact_path, allow_pickle=False) as artifact:
            schema_version = str(artifact["schema_version"][0])
            if schema_version not in SUPPORTED_MODEL_SCHEMAS:
                raise ValueError(
                    f"Unsupported artifact schema {schema_version!r}; "
                    f"expected one of {sorted(SUPPORTED_MODEL_SCHEMAS)!r}"
                )
            return cls(
                artifact["log_feature_probability"],
                artifact["log_class_prior"],
                temperature=float(artifact["temperature"][0]),
                final_turn_weight=(
                    float(artifact["final_turn_weight"][0])
                    if "final_turn_weight" in artifact
                    else 0.0
                ),
                vectorizer_version=str(artifact["vectorizer_version"][0]),
                metadata=json.loads(str(artifact["metadata_json"][0])),
            )
