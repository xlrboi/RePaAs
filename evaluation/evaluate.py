"""RAG evaluation pipeline built on DeepEval.

Run with:  uv run python evaluation/evaluate.py
(or:       make eval)
"""

from __future__ import annotations
from functools import lru_cache

import json
import sys
from pathlib import Path
from uuid import uuid4

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage

from folder.config import get_settings
from folder.core.graph import build_graph
from folder.logging_config import configure_logging, get_logger
from folder.services.paper_loader import load_document
from folder.services.vector_store import add_paper
from folder.services.llm import get_golden_llm
from folder.config import get_settings
from langchain_openai import AzureOpenAIEmbeddings


sys.stdout.reconfigure(encoding="utf-8")

sys.stderr.reconfigure(encoding="utf-8")

configure_logging()
log = get_logger("evaluation")

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = REPO_ROOT / "data" / "sample_documents" / "1603.02754v3.pdf"
GOLDENS_FILE = Path(__file__).resolve().parent / "goldens.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "eval_results.json"
EVAL_CHECKPOINT_DB = "eval_checkpoints.db"

MAX_CONTEXTS = 5
GOLDENS_PER_CONTEXT = 2
METRIC_THRESHOLD = 0.7

from deepeval.models import AzureOpenAIModel, AzureOpenAIEmbeddingModel

@lru_cache
def _get_golden_embeddings():
    settings = get_settings()
    return AzureOpenAIEmbeddingModel(
        deployment_name=settings.azure_openai_embedding_deployment,
        base_url=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )

@lru_cache
def _get_golden_deepeval_llm():
    settings = get_settings()
    return AzureOpenAIModel(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        base_url=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_api_key,
    )


def generate_goldens() -> list[dict]:
    synthesizer = Synthesizer(model=_get_golden_deepeval_llm())
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(PDF_PATH)],
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            embedder=_get_golden_embeddings(),
            critic_model=_get_golden_deepeval_llm(),
            max_contexts_per_document=MAX_CONTEXTS,
        ),
    )
    pairs = [
        {"input": g.input, "expected_output": g.expected_output}
        for g in goldens
        if g.input and g.expected_output
    ]
    GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    return pairs


def load_goldens() -> list[dict]:
    return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "query": query,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def main() -> None:
    settings = get_settings()
    pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()

    docs = load_document(str(PDF_PATH))
    graph = build_graph(db_path=EVAL_CHECKPOINT_DB)

    metrics = [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=_get_golden_deepeval_llm()),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=_get_golden_deepeval_llm()),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=_get_golden_deepeval_llm()),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=_get_golden_deepeval_llm()),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=_get_golden_deepeval_llm()),
    ]

    test_cases = []
    for pair in pairs:
        session_id = f"evaluation_session_{uuid4()}"
        add_paper(docs, session_id)

        query = pair["input"] + " as per the report in knowledge base"
        answer, retrieval_context = run_rag_query(graph, query, session_id)
        test_cases.append(
            LLMTestCase(
                input=pair["input"],
                actual_output=answer,
                expected_output=pair["expected_output"],
                retrieval_context=retrieval_context,
            )
        )

    results = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
    )

    summary = []
    for test_result in results.test_results:
        summary.append(
            {
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "success": test_result.success,
                "metrics": [
                    {"name": m.name, "score": m.score, "passed": m.success, "reason": m.reason}
                    for m in test_result.metrics_data
                ],
            }
        )

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results saved to %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
