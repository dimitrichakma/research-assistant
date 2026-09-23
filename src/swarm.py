from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_typesafe import Noul, TypeSafeClassifier
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from extract import SYSTEM_PROMPT

# Shared across all four swarm nodes below, plus Stage 2's report-generator
# nodes later - the exact repeated-call-site case that earns a
# ChatPromptTemplate (see the earlier discussion in this project): same
# SystemMessage + HumanMessage shape, every call site, only `text` and
# `instruction` differ. SYSTEM_PROMPT is baked in directly (not passed as
# a template variable) since it never varies call to call.
ASK_LLM_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{text}\n\n{instruction}"),
])


class PaperState(TypedDict):
    text: str
    visuals: list
    needed_sections: list
    jargon: str
    analogy: str
    math: str
    quiz: str
    guesses: dict


def build_graph(llm, classifier: TypeSafeClassifier, checkpointer):
    """llm, classifier, and checkpointer are all passed in, not
    constructed here - same rule as every other LLM/classifier call in
    this project. Section 9's BYOK swap and the local-vs-public
    checkpointer split both depend on this function never building its
    own instances.
    """

    def ask_llm(text, instruction):
        messages = ASK_LLM_TEMPLATE.invoke({"text": text, "instruction": instruction})
        content = llm.invoke(messages).content
        # With reasoning_effort set, content is a list of blocks (thinking
        # + text), not a plain string - same issue confirmed live in
        # tutor.py's ask_about_paper(). Every swarm node routes through
        # this one helper, so the fix belongs here, not per-node.
        if isinstance(content, list):
            return "".join(block["text"] for block in content if block.get("type") == "text")
        return content

    def supervisor_node(state):
        needed = []
        if any(v["kind"] == "equation" for v in state["visuals"]):  # free rule,
            needed.append("math")                                    # no LLM call

        result = classifier.invoke({
            "state": {"paper_text": state["text"]},
            "questions": {
                "jargon": Noul(instructions=
                    "Does this paper need jargon/term definitions for a beginner reader?"),
                "analogy": Noul(instructions=
                    "Would an ELI5 real-world analogy help explain this paper's core idea?"),
            },
        })
        if result.nouls["jargon"].noul > 0.5:
            needed.append("jargon")
        if result.nouls["analogy"].noul > 0.5:
            needed.append("analogy")
        return {"needed_sections": needed}

    def route_to_nodes(state):
        return [Send(s, state) for s in state["needed_sections"]]

    def jargon_node(state):
        return {"jargon": ask_llm(state["text"], "List domain terms + defs")}

    def analogy_node(state):
        return {"analogy": ask_llm(state["text"], "Give an ELI5 analogy")}

    def math_node(state):
        return {"math": ask_llm(state["text"], "Explain equations in plain English")}

    def quiz_node(state):
        combined = "\n\n".join(state[k] for k in ["jargon", "analogy", "math"] if k in state)
        return {"quiz": ask_llm(combined, "Write 3 comprehension-check questions")}

    def reveal_node(state):
        guesses = dict(state.get("guesses", {}))
        for section in state["needed_sections"]:
            if section not in guesses:
                guesses[section] = interrupt({
                    "section": section,
                    "prompt": f"Your guess before revealing {section}:",
                })
        return {"guesses": guesses}

    graph = StateGraph(PaperState)
    graph.add_node("supervisor", supervisor_node)
    for n, fn in [("jargon", jargon_node), ("analogy", analogy_node),
                  ("math", math_node), ("quiz", quiz_node)]:
        graph.add_node(n, fn)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_to_nodes,
                                 ["jargon", "analogy", "math"])
    for n in ["jargon", "analogy", "math"]:
        graph.add_edge(n, "quiz")

    graph.add_node("reveal", reveal_node)
    graph.add_edge("quiz", "reveal")
    graph.add_edge("reveal", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import uuid

    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
    from langchain_typesafe import TypeSafeClassifier
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    load_dotenv()
    from parsing import extract_columns, extract_visuals

    pdf_path = "docs/research_papers/paper 1.pdf"
    text = extract_columns(pdf_path)

    llm = ChatAnthropic(model="claude-opus-5-5", reasoning_effort="high", max_tokens=1500)
    classifier = TypeSafeClassifier()
    visuals = extract_visuals(pdf_path, classifier)

    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(llm, classifier, checkpointer)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        state = app.invoke({"text": text, "visuals": visuals}, config)
        print("Supervisor picked:", state["needed_sections"] or "(nothing)")

        while "__interrupt__" in state:
            pending = state["__interrupt__"][0].value
            guess = input(f"{pending['prompt']} ")
            state = app.invoke(Command(resume=guess), config)

        for key in state["needed_sections"]:
            print(f"\n=== {key.upper()} ===")
            print(state[key])
        print("\n=== QUIZ ===")
        print(state.get("quiz", "(no quiz - nothing was selected)"))
