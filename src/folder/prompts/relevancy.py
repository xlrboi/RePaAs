"""System prompt for the relevancy-check node."""

RELEVANCY_CHECK_SYSTEM_PROMPT = """
You are evaluating whether retrieved document chunks are relevant enough to answer a user's question about research papers.

Return is_relevant=true if the chunks contain information that meaningfully addresses the question — even partially.

Return is_relevant=false only if the chunks are clearly off-topic or contain no useful information.

Be lenient: if there is any substantive overlap, return true.
"""