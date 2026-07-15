"""Claim-verification node: checks whether a paper's claim has been superseded.

Uses two targeted Tavily searches (general web + ``site:arxiv.org``) rather
than the ``arxiv`` PyPI package, which was unreliable for this use case.
"""

from __future__ import annotations

from folder.constants import CLAIM_SNIPPET_CHARS, CLAIM_VERIFICATION_WEB_RESULTS, DOC_PREVIEW_CHARS
from folder.core.state import RAGState
from folder.prompts.claim_verification import CLAIM_ANALYSIS_SYSTEM_PROMPT
from folder.schemas.claims import ClaimVerificationResult
from folder.services import web_search as web_search_service
from folder.services.llm import get_main_llm


def _format_results(heading: str, results: list[dict]) -> list[str]:
    lines = [heading]
    for r in results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:DOC_PREVIEW_CHARS]}\n"
        )
    return lines


def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content

    general_results = web_search_service.search(
        f"recent research superseding: {claim[:CLAIM_SNIPPET_CHARS]}",
        max_results=CLAIM_VERIFICATION_WEB_RESULTS,
    )
    arxiv_results = web_search_service.search(
        f"site:arxiv.org {claim[:CLAIM_SNIPPET_CHARS]}",
        max_results=CLAIM_VERIFICATION_WEB_RESULTS,
    )

    lines = _format_results("=== General Web Search Results ===", general_results)
    lines += _format_results("=== arXiv Paper Search Results ===", arxiv_results)
    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_SYSTEM_PROMPT}\n\nClaim to verify:\n{claim}\n\nSearch Results:\n{context}"
    )
    llm = get_main_llm().with_structured_output(ClaimVerificationResult)
    result: ClaimVerificationResult = llm.invoke([{"role": "user", "content": prompt}])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }
