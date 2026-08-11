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


MODEL_SCHEMA_VERSION = "complexity-router-nb-v1"
VECTORIZER_VERSION = "hashed-word-ngram-v1"
DEFAULT_FEATURE_DIMENSION = 32_768
WORD_PATTERN = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+", re.IGNORECASE)
Whitespace = re.compile(r"\s+")
DecisionPolicy = Literal["argmax", "expected", "conservative"]


def normalize_prompt(text: str) -> str:
    return Whitespace.sub(" ", text.strip().lower())


def default_artifact_path() -> Path:
    return Path(
        str(files("model_router").joinpath("artifacts/complexity_router_v1.npz"))
    )


@dataclass(frozen=True)
class ComplexityPrediction:
    level: int
    expected_level: float
    confidence: float
    entropy: float
    probabilities: tuple[float, ...]
    decision_policy: str

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
        else:
            raise ValueError(f"Unknown decision policy: {decision_policy}")

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
            if schema_version != MODEL_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported artifact schema {schema_version!r}; "
                    f"expected {MODEL_SCHEMA_VERSION!r}"
                )
            return cls(
                artifact["log_feature_probability"],
                artifact["log_class_prior"],
                temperature=float(artifact["temperature"][0]),
                vectorizer_version=str(artifact["vectorizer_version"][0]),
                metadata=json.loads(str(artifact["metadata_json"][0])),
            )
