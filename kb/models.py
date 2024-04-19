from typing import TypedDict, Callable, Protocol

import torch


class Entity(TypedDict):
    title: str
    url: str
    summary: str


class Relation(TypedDict):
    head: str

    implicit: bool
    type: str | None
    embedding: torch.Tensor | None

    tail: str
    meta: dict


class ExternalKnowledgeBase(Protocol):
    def get_entity(self, entity_candidate: str) -> Entity | None: ...


EmbeddingsFunction = Callable[[str], torch.Tensor]

SimilarityFunction = Callable[[str, str], float]
