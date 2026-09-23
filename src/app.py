import os
import sqlite3
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_typesafe import TypeSafeClassifier
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from parsing import extract_blocks, extract_columns, extract_visuals
from swarm import build_graph
from tutor import ask_about_paper

# Streamlit doesn't auto-load .env the way some frameworks do - missed
# this when first writing this file, confirmed by actually running the
# app: TypeSafeClassifier() failed with "API key required" even though
# TYPESAFE_API_KEY is genuinely in .env, because nothing had loaded it
# into the process environment yet.
load_dotenv()

st.set_page_config(page_title="Research Paper Assistant", page_icon="📖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Title + subtitle - centered on the page */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0 !important;
    text-align: center;
}
.rpa-subtitle {
    color: rgba(250, 250, 250, 0.6);
    font-size: 1.05rem;
    margin-top: 0.25rem;
    margin-bottom: 1.75rem;
    text-align: center;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid rgba(120, 170, 255, 0.4);
    background: linear-gradient(135deg, #3b6fd6, #5b8def);
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
    transition: transform 0.05s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
    box-shadow: 0 4px 14px rgba(91, 141, 239, 0.35);
    border-color: rgba(120, 170, 255, 0.7);
}
.stButton > button:active { transform: scale(0.98); }

/* Text inputs / password field */
.stTextInput input {
    border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px;
    border: 1.5px dashed rgba(120, 170, 255, 0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 1.02rem;
    font-weight: 600;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Status / expander boxes */
[data-testid="stExpander"], [data-testid="stStatusWidget"] {
    border-radius: 10px !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
    background: rgba(255, 255, 255, 0.03);
}

/* Study Guide reveal columns as cards - targets st.container(border=True)'s
   real wrapper element, not a hand-rolled div (that approach was tried and
   rejected: st.markdown() calls each render as their own separate block in
   Streamlit, so "opening" and "closing" a div across two separate
   st.markdown() calls doesn't actually wrap anything in between - verified
   by testing it in isolation, it produced empty floating boxes with the
   real content sitting unstyled beneath them). */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.035);
}
</style>
""", unsafe_allow_html=True)

st.title("📖 Research Paper Reading Assistant")
st.markdown(
    '<div class="rpa-subtitle">Upload a paper, build your own understanding first, '
    'then check it against grounded, cited explanations.</div>',
    unsafe_allow_html=True,
)


def get_llm(api_key):
    if api_key.startswith("sk-ant-"):
        return ChatAnthropic(model="claude-opus-4-5-20251101", api_key=api_key)
    elif api_key.startswith("sk-"):
        # OpenAI model name not verified against current OpenAI docs - this
        # branch is untested (no OpenAI key available while building this),
        # unlike everything else in this project. Check the current model
        # name before relying on this path.
        return ChatOpenAI(model="gpt-5.1", api_key=api_key)
    else:
        st.error("Doesn't look like a Claude or OpenAI key")
        st.stop()


# API key entry lives in the main page, centered, not tucked into the
# sidebar - the sidebar's only job here was this one field, so it's
# dropped entirely rather than left holding nothing.
if "api_key" not in st.session_state:
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.subheader("API Key")
        entered_key = st.text_input("Paste your Claude or OpenAI key", type="password")
        if not entered_key:
            st.info("Paste an API key to get started.")
            st.stop()
        st.session_state.api_key = entered_key
        st.rerun()
api_key = st.session_state.api_key

if not api_key:
    st.sidebar.info("Paste an API key to get started.")
    st.stop()
llm = get_llm(api_key)

# TypeSafeClassifier uses its own TYPESAFE_API_KEY (.env), never the
# visitor's pasted BYOK key - it isn't the swappable provider BYOK targets,
# it's a fixed judgment layer the app itself owns. Cached in session_state
# so it's built once per session, not reconstructed every rerun.
if "classifier" not in st.session_state:
    st.session_state.classifier = TypeSafeClassifier()
classifier = st.session_state.classifier

# Local-vs-public checkpointer split, same reasoning as data/library.db
# (Section 10): a shared persistent checkpointer on a public deploy would
# leak one visitor's paused study-guide state into another's. Built with
# a raw sqlite3.Connection (check_same_thread=False), not the
# from_conn_string context-manager form - Streamlit reruns the whole
# script on every interaction, so the checkpointer has to survive across
# many separate reruns within a session, not just one `with` block.
# Verified: dropping and reconnecting this exact way mid-session still
# resumes a paused graph correctly, so a Streamlit rerun (much lighter
# than a full reconnect) is safe too.
if "checkpointer" not in st.session_state:
    if os.getenv("PUBLIC_DEPLOY"):
        st.session_state.checkpointer = MemorySaver()
    else:
        conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
        st.session_state.checkpointer = SqliteSaver(conn)
checkpointer = st.session_state.checkpointer

app = build_graph(llm, classifier, checkpointer)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Upload and parse
if "paper_text" not in st.session_state:
    uploaded = st.file_uploader("Upload a paper (PDF)", type="pdf")
    if uploaded:
        with st.status("Reading the paper...", expanded=True) as status:
            st.write("Saving upload...")
            # extract_columns()/extract_visuals()/extract_blocks() all take
            # a path, not a file-like object - write the upload to a real
            # temp file rather than change every Level 1 function's
            # signature to also accept a stream.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            st.write("Extracting text (Level 1)...")
            text = extract_columns(tmp_path)
            st.write("Extracting block structure (Level 1)...")
            blocks = extract_blocks(tmp_path)
            st.write("Cropping figures and equations (Level 1)...")
            visuals = extract_visuals(tmp_path, classifier, output_dir=tempfile.mkdtemp())
            status.update(label="Ready", state="complete")

        os.unlink(tmp_path)
        st.session_state.paper_text = text
        st.session_state.blocks = blocks
        st.session_state.visuals = visuals
        st.rerun()
    st.stop()  # nothing below runs until a paper is loaded

# The two tabs
ask_tab, guide_tab = st.tabs(["Ask Questions", "Study Guide"])

# Ask Questions - chat, with the tool_log shown, not hidden
with ask_tab:
    if "chat" not in st.session_state:
        st.session_state.chat = []  # list of {question, answer, tool_log}

    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["tool_log"]:
                with st.expander(f"{len(turn['tool_log'])} tool call(s) used"):
                    for call in turn["tool_log"]:
                        st.code(f"{call['tool']}({call['input']})", language=None)

    question = st.chat_input("Ask something about this paper")
    if question:
        with st.spinner("Reading..."):
            result = ask_about_paper(llm, classifier, st.session_state.paper_text,
                                      st.session_state.blocks, question)
        st.session_state.chat.append({"question": question,
                                       "answer": result["answer"], "tool_log": result["tool_log"]})
        st.rerun()

# Study Guide - the supervisor's decision, then the genuine gated reveal
with guide_tab:
    if "guide_state" not in st.session_state:
        with st.spinner("Deciding what this paper needs..."):
            st.session_state.guide_state = app.invoke(
                {"text": st.session_state.paper_text, "visuals": st.session_state.visuals}, config)

    state = st.session_state.guide_state
    labels = {"jargon": "Jargon Buster", "analogy": "ELI5 Analogy", "math": "Math, in English"}

    if not state["needed_sections"]:
        st.caption("This paper's supervisor picked: nothing - no jargon, analogy, or math needed.")
    else:
        st.caption("This paper's supervisor picked: " +
                   ", ".join(labels[k] for k in state["needed_sections"]))

    if "__interrupt__" in state:  # paused, waiting on a guess
        pending = state["__interrupt__"][0].value
        guess = st.text_input(pending["prompt"], key=pending["section"])
        if guess and st.button("Submit guess"):
            st.session_state.guide_state = app.invoke(Command(resume=guess), config)
            st.rerun()
    elif state["needed_sections"]:  # every section answered - fully revealed
        cols = st.columns(len(state["needed_sections"]))
        for col, key in zip(cols, state["needed_sections"]):
            with col, st.container(border=True):
                st.subheader(labels[key])
                st.write(state[key])
        st.subheader("Comprehension Quiz")
        with st.container(border=True):
            st.write(state["quiz"])
