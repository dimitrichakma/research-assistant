import re

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_typesafe import Choice, Noul, TypeSafeClassifier
from langgraph.errors import GraphRecursionError
from rank_bm25 import BM25Okapi

from extract import SYSTEM_PROMPT
from parsing import explain_visual


def _detect_headings(blocks, classifier, chunk_size=150):
    """Classify blocks as section headings, in chunks.

    One call covering the whole document's blocks can exceed TypeSafe's
    per-request limit, the same failure shape as Level 1's
    extract_visuals() batching problem (Section 4) - confirmed
    empirically: LORA.pdf's 628 blocks in a single call raised a 400
    Bad Request (TypeSafeBadRequestError), even though this only asks
    one question per block (extract_visuals() asks two). Binary-searched
    the actual boundary: 500 blocks succeeded, 628 failed - chunk_size
    here is set well under that confirmed-safe range, not right at the
    edge, since the real limit is likely content-length-dependent, not
    a fixed block count, and a different paper's blocks could be denser.

    This bug was silently hidden until directly tested: get_section()
    wraps its whole body in try/except and returns the error as text,
    so a failure here previously just looked like "no headings found"
    to both the agent and to a human skimming the final answer - the
    agent quietly routed around it using find_quote instead, and 3
    earlier test runs "passed" without anyone noticing get_section was
    actually failing every single call.
    """
    headings = []
    for start in range(0, len(blocks), chunk_size):
        chunk = blocks[start:start + chunk_size]
        heading_qs = {
            f"h{i}": Noul(instructions=f"Is the text in `h{i}` a section heading, not body text?")
            for i in range(len(chunk))
        }
        flags = classifier.invoke({
            "state": {f"h{i}": b[4] for i, b in enumerate(chunk)},
            "questions": heading_qs,
        })
        headings.extend(chunk[i] for i in range(len(chunk)) if flags.nouls[f"h{i}"].noul > 0.5)
    return headings


def _build_tools(llm, paper_text, blocks, classifier: TypeSafeClassifier):
    """Build the three ReAct tools, bound to one paper and one classifier
    via closure rather than module-level globals set by a set_paper()
    call. Each call to ask_about_paper() gets its own independent set of
    tools - two papers (or two concurrent sessions, once Level 4.5's UI
    exists) never share mutable state, unlike a global PAPER_TEXT/_bm25
    that a second call could silently overwrite mid-question.
    """
    sentences = re.split(r"(?<=[.!?])\s+", paper_text)
    bm25 = BM25Okapi([s.lower().split() for s in sentences])

    @tool
    def get_section(section_name: str) -> str:
        """Return one named section of the paper (Abstract, Methods, Results,
        Discussion, Conclusion) instead of the whole paper - use this for
        long papers where you only need one part."""
        try:
            headings = _detect_headings(blocks, classifier)
            if not headings:
                return ("No headings detected. Try Abstract, Methods, Results, "
                        "Discussion, or Conclusion.")

            choice = classifier.invoke({
                "state": {"target": section_name,
                          **{f"c{i}": h[4] for i, h in enumerate(headings)}},
                "questions": {"match": Choice(
                    instructions="Which heading matches the target section, if any?",
                    criteria={f"c{i}": h[4][:60] for i, h in enumerate(headings)}
                    | {"none": "no match"},
                )},
            })
            picked = choice.choices["match"].choice
            if picked == "none":
                return f"No section named '{section_name}' found."
            idx = int(picked[1:])
            heading_text = headings[idx][4]
            start = paper_text.find(heading_text)
            if start == -1:
                return "Found a heading but couldn't locate it in the text."
            return paper_text[start:start + 3000]
        except Exception as e:
            return f"Section lookup failed: {e}"

    @tool
    def find_quote(claim: str) -> str:
        """Find the most relevant exact sentence in the paper for a claim -
        use before answering, to ground the answer in a real quote rather
        than a paraphrase."""
        try:
            scores = bm25.get_scores(claim.lower().split())
            top_k = scores.argsort()[-5:][::-1]
            candidates = {f"s{i}": sentences[idx] for i, idx in enumerate(top_k)}

            result = classifier.invoke({
                "state": {"claim": claim, **candidates},
                "questions": {
                    f"supports_{k}": Noul(
                        instructions=f"Does the passage in `{k}` actually state the "
                                     f"thing being claimed, not just share vocabulary with it?")
                    for k in candidates
                },
            })
            best_key = max(candidates, key=lambda k: result.nouls[f"supports_{k}"].noul)
            best = candidates[best_key]
            return best.strip() or "No close match found - try rephrasing the claim."
        except Exception as e:
            return f"Quote search failed: {e}"

    @tool
    def explain_figure(figure_path: str, kind: str = "figure") -> str:
        """Get a vision-based explanation of one cropped figure/equation image
        (paths come from Level 1's extract_visuals()) - use only when the
        question is actually about a specific chart or equation."""
        try:
            return explain_visual(llm, figure_path, kind)
        except Exception as e:
            return f"Couldn't read that image: {e}. Check the path from extract_visuals()."

    return [get_section, find_quote, explain_figure]


def ask_about_paper(llm, classifier, paper_text, blocks, question, your_guess=None):
    """llm and classifier are both passed in, not constructed here - same
    reason as Level 2's extract_paper(llm, text): Section 9's BYOK swap
    needs one place to change the provider, not a hardcoded model buried
    in every function. classifier follows the identical rule.

    paper_text and blocks come from Level 1's extract_columns() and
    extract_blocks() respectively - extracted once per paper, passed in
    here so repeated questions about the same paper don't re-parse the
    PDF on every call.

    your_guess is optional - a CLI/UI can pass one to compare against;
    leaving it out is plain Q&A.
    """
    tools = _build_tools(llm, paper_text, blocks, classifier)
    agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=
                f"Question: {question}\n\n"
                f"Use get_section / find_quote / explain_figure to find the "
                f"answer - don't guess.")]},
            # a hard cap so a confusing question can't run unbounded tool
            # calls - verified empirically at 9 tool calls in one real run
            # with no GraphRecursionError, so the "2 graph steps per tool
            # call" estimate undersells the real capacity; treat 15 as a
            # reasonable cap, not a precisely-known call count
            config={"recursion_limit": 15},
        )
    except GraphRecursionError:
        # confirmed this actually happens on a real question, real paper -
        # recursion_limit alone isn't a graceful cap, agent.invoke() just
        # raises once it's hit. Without this, the limit only protects your
        # API bill, not the caller - the whole function crashes instead of
        # returning something usable.
        return {
            "answer": ("I wasn't able to find a confident answer within the "
                       "tool-call budget for this question - try asking "
                       "something more specific."),
            "tool_log": [],
            "your_guess": your_guess,
        }
    tool_log = [
        {"tool": call["name"], "input": call["args"]}
        for msg in result["messages"] if getattr(msg, "tool_calls", None)
        for call in msg.tool_calls
    ]
    answer = result["messages"][-1].content
    return {"answer": answer, "tool_log": tool_log, "your_guess": your_guess}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from langchain_typesafe import TypeSafeClassifier

    load_dotenv()
    from parsing import extract_blocks, extract_columns

    pdf_path = "docs/research_papers/paper 1.pdf"
    text = extract_columns(pdf_path)
    blocks = extract_blocks(pdf_path)

    llm = ChatAnthropic(model="claude-opus-4-5-20251101", max_tokens=1500)
    classifier = TypeSafeClassifier()

    question = input("Ask something about this paper: ")
    guess = input("Your guess: ")
    result = ask_about_paper(llm, classifier, text, blocks, question, your_guess=guess)
    print(f"\nYour guess:  {result['your_guess']}")
    print(f"AI answer:   {result['answer']}")
    print(f"Tools used:  {[t['tool'] for t in result['tool_log']]}")
