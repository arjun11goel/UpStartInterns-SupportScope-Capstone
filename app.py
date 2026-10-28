from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "src"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from supportscope.assistant import DEFAULT_MODEL, answer_question
st.set_page_config(page_title="SupportScope AI", page_icon="🧭", layout="centered")


def get_api_key() -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def get_model() -> str:
    model = os.getenv("GROQ_MODEL")
    if model:
        return model
    try:
        return st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    except Exception:
        return DEFAULT_MODEL


st.title("🧭 SupportScope AI")
st.caption("A grounded customer-support assistant for the fictional Northstar Store")
st.markdown(
    "Ask about shipping, returns, refunds, cancellations, damaged items, warranties, "
    "accepted payments, or contacting support. Responses are based on a fixed policy knowledge base."
)

with st.sidebar:
    st.header("Project controls")
    st.write("**Model:** `" + get_model() + "`")
    st.write("**Retrieval:** TF-IDF, top 2 policy excerpts")
    st.write("**Guardrails:** unsafe-input check + out-of-scope refusal")
    st.info("This is a portfolio demo using fictional store policies. It cannot access or modify real orders.")
    st.divider()
    st.subheader("Session usage")
    st.metric("Requests", st.session_state.get("request_count", 0))
    st.metric("Total tokens", st.session_state.get("session_tokens", 0))
    st.metric("Estimated API cost", f"${st.session_state.get('session_cost', 0.0):.6f}")
    st.caption("Cost is estimated at published token rates for comparison. Free-tier billing and limits depend on provider account settings.")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.request_count = 0
        st.session_state.session_tokens = 0
        st.session_state.session_cost = 0.0
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0

with st.expander("Try sample questions"):
    st.markdown(
        "- How long does standard shipping take?\n"
        "- Can I return an unopened item?\n"
        "- How long do approved refunds take?\n"
        "- My item arrived damaged. What should I do?\n"
        "- Explain recursion in Python. (should be refused)"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Ask a store-policy question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking scope and retrieving policy evidence…"):
            result = answer_question(prompt, api_key=get_api_key(), model=get_model())
        st.markdown(result.answer)
        if result.source_ids:
            with st.expander("Retrieved evidence and telemetry"):
                st.write("**Source IDs:** " + ", ".join(result.source_ids))
                st.write(f"**Maximum retrieval score:** {result.max_retrieval_score:.4f}")
                st.write(f"**Status:** `{result.status}`")
                st.write(f"**Latency:** {result.latency_seconds:.3f} seconds")
                st.write(f"**Token usage:** {result.input_tokens} input + {result.output_tokens} output = {result.total_tokens} total")
                st.write(f"**Estimated cost:** ${result.estimated_cost_usd:.8f}")
                st.write(f"**Model:** `{result.model}`")
        else:
            st.caption(f"Status: {result.status} · Latency: {result.latency_seconds:.3f}s · Model calls: 0 for blocked requests")

    meta = (
        f"Status: {result.status} · Tokens: {result.total_tokens} · "
        f"Estimated cost: ${result.estimated_cost_usd:.6f} · Latency: {result.latency_seconds:.2f}s"
    )
    st.session_state.messages.append({"role": "assistant", "content": result.answer, "meta": meta})
    st.session_state.request_count += 1
    st.session_state.session_tokens += result.total_tokens
    st.session_state.session_cost += result.estimated_cost_usd

    st.rerun()
