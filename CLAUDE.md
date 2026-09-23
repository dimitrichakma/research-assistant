# Project Overview
A personal research-paper reading assistant. A single-paper study-guide
generator (jargon, analogy, math explanation, comprehension quiz) that's
meant to grow into a multi-paper research companion once that's actually
needed. Built level by level against a local design roadmap
(docs/research_assistant_roadmap.pdf - gitignored, local reference only,
not part of the shipped repo). The goal is to build the user's own
reading skill, not replace it: every AI explanation must cite its
source, and nothing is revealed until the user engages first (a guess,
or a quiz).

All five core levels are built and tested against real papers:
Level 1 (src/parsing.py), Level 2 (src/extract.py), Level 3
(src/tutor.py), Level 4 (src/swarm.py), Level 4.5 (src/app.py).

# Tech Stack
- LLM: Claude via LangChain (`ChatAnthropic`) throughout every level,
  not the raw Anthropic SDK - so no LLM call is bound to one provider's
  SDK. Level 4.5's bring-your-own-key swap is just constructing a
  different ChatModel (`ChatAnthropic` vs. `ChatOpenAI`) - no
  provider-specific branching anywhere downstream. The OpenAI branch of
  `get_llm()` (src/app.py) is unverified - no OpenAI key was available
  while building it, unlike everything else in this project.
- Agent orchestration: Level 3's ReAct agent uses LangChain's
  `create_agent` (`langchain.agents`) - migrated 2026-09-23 from
  LangGraph's `create_react_agent`, which is deprecated in favor of it.
  Verified by actually migrating and re-testing against real papers, not
  by reading the deprecation notice: only real differences were the
  import path and the system-prompt argument renaming from `prompt` to
  `system_prompt`; message shape, `.tool_calls`, and `recursion_limit`
  behavior (including `GraphRecursionError`) were identical. `langchain`
  is a real dependency now, not just `langchain-core`/`langchain-anthropic`.
  Level 4's `StateGraph` - a supervisor decides ONCE which nodes run
  (real delegation via `Send()`), then fixed parallel execution; not an
  iterating loop, so no `recursion_limit` needed there. `reveal_node`'s
  `interrupt()` is a third, distinct category - a real
  `interrupt()`/`Command(resume=...)` pause per section, gated on a
  HUMAN guess, not model iteration or a single decision.
- Prompting: one shared `SYSTEM_PROMPT` (src/extract.py - the first
  place in the build order that actually calls an LLM) wired directly
  into its three call sites: Level 2's extraction call, Level 3's agent
  via `create_agent`'s `system_prompt=` argument, and Level 4's shared
  `ask_llm()` helper (covers all four swarm nodes in one place) - never
  restated inline per call. Level 4's `ask_llm()` is built on a
  `ChatPromptTemplate` (`ASK_LLM_TEMPLATE` in src/swarm.py), not plain
  message construction - the one place in this project where that's
  actually earned: it's reused across 5+ call sites with the same
  `SystemMessage` + `HumanMessage` shape. Level 2 and Level 3's single
  call sites don't use a template - not enough repetition yet to be
  worth it there.
- Context: Level 2 and Level 4 always receive the WHOLE paper text,
  never a summary - citations require the actual source text;
  summarizing first breaks the citations feature entirely. Level 3
  instead pulls only the section/quote it needs, via its own tools.
- PDF parsing: PyMuPDF (`fitz`) in src/parsing.py -
  `extract_columns()` (text, column-order corrected),
  `extract_blocks()` (raw block structure, for Level 3's `get_section()`),
  `extract_visuals()` (figure/equation cropping, TypeSafe-classified -
  returns `{"path": ..., "kind": "figure"|"equation"}` dicts, not bare
  paths, since Level 4's supervisor rule check needs the kind tag),
  `explain_visual(llm, image_path, kind)` (vision-based explanation, no
  OCR needed). `_merge_fragment_blocks()` groups PyMuPDF's per-symbol
  equation fragments before classification (character count, not pixel
  width, separates fragments from real paragraphs - verified empirically,
  width alone didn't have a reliable cutoff). `_crop_top_for_caption()`
  finds a figure's real vertical extent by walking upward through short
  (label) blocks above the caption, instead of a fixed-height guess.
- Judgment calls (not generation): `langchain-typesafe`'s
  `TypeSafeClassifier` (`Choice`/`Noul`), replacing what used to be
  regex or free-text-then-parse - Level 1's figure/equation block
  classification, Level 3's `get_section()` heading match and
  `find_quote()`'s BM25 candidate rerank, and both supervisors' (Level 4,
  Stage 2) section routing. Never used for citation text itself -
  citations come from Claude's own citation feature reading the paper's
  actual source text. `langchain-typesafe` was chosen deliberately over
  the raw `typesafe-sdk` - it's a LangChain `Runnable`, composing
  naturally inside LangGraph nodes, even though it's alpha
  (`0.0.1a3` at time of writing).
- Search: `find_quote()` (Level 3) ranks candidate sentences with
  `rank_bm25` (`BM25Okapi`), in-memory only, rebuilt per paper via
  closures in `_build_tools()` (src/tutor.py) - not module-level
  globals, so two papers (or two concurrent sessions) never share
  mutable state. TypeSafe reranks BM25's top-5 candidates by whether
  each one actually supports the claim, not just shares vocabulary.
  Stage 2's cross-paper keyword search (not yet built) would apply the
  same BM25 ranking across a persisted library, via SQLite FTS5's
  `bm25()`.
- Batching TypeSafe calls: a single `classifier.invoke()` covering too
  much content raises a 400 (`TypeSafeBadRequestError`), not a partial
  failure - confirmed twice, independently: Level 1's `extract_visuals()`
  batching an entire document's blocks into one call, and Level 3's
  `get_section()` doing the same for its heading-detection pass. Both
  fixed by chunking (per-page for Level 1, `chunk_size=150` for Level 3's
  `_detect_headings()`) - sized well under the empirically-confirmed-safe
  range, not right at its edge, since the real limit is content-length-
  dependent, not a fixed count. `get_section()`'s failure was invisible
  until tested directly: its own `try/except` disguised the 400 as
  "no headings found," and the agent quietly routed around it with
  `find_quote` instead across multiple "passing" test runs.
- Checkpointer: `SqliteSaver` locally, `MemorySaver` on a public deploy -
  a shared persistent checkpointer would leak one visitor's paused
  session into another's. Built via `SqliteSaver(sqlite3.Connection)`
  directly (`check_same_thread=False`), not the `from_conn_string`
  context-manager form - Streamlit reruns the whole script on every
  interaction, so the checkpointer has to survive across many separate
  reruns within a session, not just one `with` block. Verified: dropping
  and fully reconnecting the SQLite connection mid-session (harsher than
  a Streamlit rerun) still resumes a paused graph correctly.
- Structured output: Pydantic + Claude's native citations feature -
  but NOT in the same call. Verified empirically: `with_structured_output()`
  and `citations: {"enabled": True}` together succeed without error, but
  the underlying response is a plain tool-call with no citations field
  at all - `with_structured_output()` forces tool-calling, and Claude's
  citation mechanism only annotates generated text. `extract_paper()`
  (src/extract.py) uses two calls instead: one for the schema, one
  separate citations-enabled call that restates the same claims so they
  get real, verified grounding.
- UI: Streamlit (src/app.py) - not Next.js; personal, single-user tool,
  so a build step and multi-user auth buy nothing here. Streamlit does
  NOT auto-load `.env` - missed this when first writing app.py, confirmed
  by actually running it (`TypeSafeClassifier()` failed with "API key
  required" despite a genuinely-set `TYPESAFE_API_KEY`) until
  `load_dotenv()` was added explicitly. API key entry lives centered in
  the main page, not the sidebar - the sidebar had no other job, so it's
  dropped entirely. Custom CSS for card-style layout targets
  `st.container(border=True)`'s real wrapper element
  (`[data-testid="stVerticalBlockBorderWrapper"]`), not a hand-rolled
  `<div>` - that approach was tried and rejected: `st.markdown()` calls
  each render as their own separate block in Streamlit, so "opening" and
  "closing" a div across two separate calls doesn't wrap anything
  between them (verified in isolation before it was trusted).
- Storage: SQLite, LOCAL-ONLY - no vector database until Stage 3 of the
  growth path is actually reached. `data/library.db` (Stage 2, cross-
  paper library) is planned, not built - don't add it ahead of its own
  trigger (see Rules).

# Project Structure
```
src/parsing.py       Level 1 - PDF text extraction (column-order
                      corrected), block extraction, figure/equation
                      cropping (TypeSafe-classified), vision explanation
src/extract.py        Level 2 - structured, citation-grounded extraction
                      (two calls - see Structured output above);
                      SYSTEM_PROMPT defined here
src/tutor.py          Level 3 - ReAct agent: get_section (chunked
                      TypeSafe heading match) / find_quote (BM25 +
                      TypeSafe rerank) / explain_figure tools, built
                      per-call via closures; recursion_limit + caught
                      GraphRecursionError; tool_log
src/swarm.py          Level 4 - supervisor-gated study guide (Send()
                      picks jargon/analogy per paper via TypeSafe Nouls;
                      math gated by a free rule); real interrupt()-based
                      reveal loop; ASK_LLM_TEMPLATE (ChatPromptTemplate)
src/app.py            Level 4.5 - Streamlit UI: two tabs (Ask Questions -
                      chat + visible tool_log; Study Guide - supervisor's
                      picks shown, interrupt-aware reveal loop), thread_id
                      per session, bring-your-own API key (centered,
                      Claude or OpenAI), custom CSS
checkpoints.db        SQLite - LangGraph checkpointer (reveal_node's
                      paused state) - LOCAL USE ONLY, gitignored
docs/                 architecture diagrams (agent_graph.png,
                      react_graph.png - committed) and design notes
                      (roadmap/notes PDFs - gitignored, local only)
docs/research_papers/ downloaded test PDFs - gitignored (copyrighted)
figures/              extract_visuals() output - gitignored (crops of
                      copyrighted papers)
README.md             portfolio-facing project writeup
```
Not yet built (Stage 2+, no trigger hit yet): `data/library.db` - see
Workflows and Rules.

# Workflows
Build and test one level at a time, in order - each level's "done
state" was verified against 3+ real, different papers before moving on,
not just unit-tested in isolation. Growing beyond one paper (Stage 2+,
a persisted cross-paper library) is a second, independent axis from the
Level 1-4.5 build above and hasn't been started - it scales
independently and shouldn't be built ahead of its own trigger (a real
personal library forming, keyword search starting to fail, wanting to
search by what a figure looked like - not a guessed paper count).

# Rules
- Never add retrieval infrastructure (vector DB, multimodal search, or
  Stage 2's `data/library.db`) until its specific trigger in the
  growth-path staging is actually hit - no speculative infrastructure.
- `data/library.db` (once built) is local-only. The public deployment
  never reads or writes it - it stays upload-in-memory-only, so no
  visitor's data can leak into another's on a shared file. Same rule for
  the LangGraph checkpointer: `SqliteSaver` (`checkpoints.db`) locally,
  `MemorySaver` on the public deploy - never a shared persistent
  checkpointer serving multiple public visitors.
- Every AI-generated explanation must carry a citation back to the exact
  source sentence.
- No swarm node's output is shown to the user without a gating step
  first (a typed guess, or answering the quiz) - `reveal_node`'s
  `interrupt()` is a real backend pause, not a UI-only trick; verify any
  future change to this flow still halts actual graph execution, not
  just hides a widget.
- Keep grounding rules in the one shared `SYSTEM_PROMPT`, not restated
  per call - Level 2, Level 3, and Level 4 all reuse it.
- Level 3's `create_agent()` always gets a `recursion_limit` in its
  `invoke()` config, AND the caller catches `GraphRecursionError` around
  that call - the limit alone isn't a graceful cap, it just raises once
  hit; confirmed this crashes the caller for real if uncaught.
- Every Level 3 tool catches its own exceptions and returns the error as
  text - but verify what's actually happening on a failure path, not
  just that it doesn't crash: `get_section()`'s own error handling once
  disguised a real, systemic failure (batching over TypeSafe's request
  limit) as an innocuous "no headings found" for multiple test runs in a
  row.
- Level 4's and Stage 2's supervisors each decide ONCE, up front, which
  nodes/sections to run - that single routing decision needs no
  `recursion_limit` or tool-log discipline; those only apply to Level 3's
  actual iterating loop.
- Every LLM call goes through a LangChain ChatModel instance
  (`ChatAnthropic` by default) - never import from `anthropic` directly.
  Critically, that instance is always passed IN as a parameter
  (`extract_paper(llm, text)`, `ask_about_paper(llm, ...)`,
  `build_graph(llm, classifier, checkpointer)`) - never constructed
  inside the function that uses it. Same rule for `TypeSafeClassifier`.
  This is what actually makes Level 4.5's bring-your-own-key swap work:
  `get_llm(api_key)` only has to change ONE line because nothing
  downstream builds its own model.
- TypeSafe questions (`Choice`/`Noul`) are for judgment only - classify,
  select, or rank - never for producing the citation text itself. Cheap
  regex/BM25 first-pass filters stay in place where they already work;
  TypeSafe adds a judgment layer on top of them, it doesn't replace the
  whole pipeline.
- When batching multiple questions into one `classifier.invoke()` call,
  size the batch well under any empirically-confirmed-safe limit, not
  right at its edge - the real ceiling is content-length-dependent, not
  a fixed count, so a different paper's denser text could fail at a
  smaller batch size than one that worked before.
- Don't trust that "no error raised" means a feature actually works -
  verify the real behavior directly. Two confirmed examples this
  project actually hit: a citations-enabled + structured-output call
  that succeeds but silently drops all citations, and a `try/except`-
  wrapped tool whose "clean" error message was hiding a real, repeatable
  failure across several passing-looking test runs.
