from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

# Shared across Level 2's extraction call, Level 3's ReAct agent (Section 6),
# and Level 4's swarm nodes (Section 7) - defined here since this is the
# first place in the build order that actually calls an LLM.
SYSTEM_PROMPT = (
    "You are a careful research-paper reading assistant. Your job across "
    "every task - extracting a summary, answering a question, writing a "
    "jargon list, an analogy, or a quiz - is the same: ground every claim "
    "in the paper's actual text, and never state something the paper "
    "doesn't actually say.\n\n"

    "Citation discipline:\n"
    "Bad: 'The study shows strong results.' - vague, no citation.\n"
    "Good: 'Accuracy improved from 71% to 84%.' - specific, citable.\n\n"

    "If you don't have enough information in what you've been shown, say "
    "so directly - don't fill the gap with general domain knowledge the "
    "paper itself doesn't state.\n\n"

    "When a tool is available to look something up (a section, an exact "
    "quote, a figure), use it before answering rather than guessing from "
    "context - a wrong guess stated confidently is worse than one more "
    "tool call.\n\n"

    "When writing an explanation - a jargon definition, an analogy, a "
    "plain-English equation walkthrough - tie it to what this specific "
    "paper says, not a generic textbook definition of the topic."
)


class Extraction(BaseModel):
    research_question: str
    method: str
    key_result: str
    limitations: str
    sample_size: str
    is_preprint: bool
    causal_or_correlational: str


_FIELD_LABELS = {
    "research_question": "Research question",
    "method": "Method",
    "key_result": "Key result",
    "limitations": "Limitations",
    "sample_size": "Sample size",
    "causal_or_correlational": "Causal or correlational",
}


def extract_paper(llm, text):
    """One structured extraction, plus a separate citations-enabled call
    for real grounding quotes.

    Anthropic's citations feature and with_structured_output() don't
    combine, despite the API call succeeding without error - verified
    empirically: a citations-enabled document passed alongside a forced
    tool call returns a plain tool_use block with no citations field at
    all. LangChain's with_structured_output() implements structured
    output via tool-calling, and Claude's citation mechanism only
    annotates generated TEXT blocks, not tool-call arguments - so
    "citations": {"enabled": True} is silently ignored the moment
    structured output is also requested. This was flagged as an open
    question in the roadmap (Section 5); it's a confirmed, real
    limitation, not a hypothetical one.

    So: one call gets the structured schema (fast, reliable, no
    citations attempted since they'd be dropped anyway). A second,
    separate citations-enabled call asks the model to restate those same
    claims, grounded in real citations this time - real char-offset
    citations only exist on generated text, so the model has to actually
    write the claims out again for Claude's citation mechanism to attach
    to them.

    Returns a dict, not just the Extraction instance: {"extraction": the
    structured Pydantic object, for anything that needs the fields
    programmatically (review schedule, filtering, Level 4's supervisor);
    "citations": the raw citations-enabled response content, a list of
    {"text": ..., "citations": [{"cited_text": ..., "start_char_index":
    ..., "end_char_index": ...}, ...]} blocks, for display - showing the
    user what the paper actually says, with real citations attached}.
    """
    structured_llm = llm.with_structured_output(Extraction)
    extraction = structured_llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "document",
             "source": {"type": "text", "media_type": "text/plain", "data": text}},
            {"type": "text", "text":
             "Extract: research question, method, key result, limitations, "
             "sample size, whether it's a preprint, and whether the finding "
             "is causal or correlational."}
        ]),
    ])

    claims = "\n".join(
        f"{i}. {_FIELD_LABELS[field]}: {getattr(extraction, field)}"
        for i, field in enumerate(_FIELD_LABELS, start=1)
    )
    citation_response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "document",
             "source": {"type": "text", "media_type": "text/plain", "data": text},
             "citations": {"enabled": True}},
            {"type": "text", "text":
             f"Restate each of the following claims about this paper, each "
             f"as its own sentence, grounded in a citation from the source "
             f"text:\n{claims}"},
        ]),
    ])

    return {"extraction": extraction, "citations": citation_response.content}
