"""
Streamlit front end for the Story Director agent.

Shows the PLAN / ACT / OBSERVE / REFLECT loop running live: every draft, every
constraint check, and the prompt-injection strategy chosen after each failure.

    streamlit run streamlit_app.py
"""

import os

import streamlit as st
import torch

from demo import CHECKPOINT_URL, download_checkpoint
from src.agent import StoryDirectorAgent, contains_word, excludes_word, min_words
from src.model import GPT, GPTConfig

st.set_page_config(page_title="Agentic SLM", page_icon="📖", layout="wide")

CHECKPOINT = "best_model_params.pt"


@st.cache_resource(show_spinner=False)
def load_model():
    """Loaded once per process; the weights are ~115MB."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig())
    trained = False
    if not os.path.exists(CHECKPOINT):
        download_checkpoint(CHECKPOINT)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        trained = True
    model.to(device).eval()

    import tiktoken
    return model, tiktoken.get_encoding("gpt2"), device, trained


st.title("Agentic SLM: The Story Director")
st.caption(
    "A small decoder-only Transformer trained from scratch on TinyStories, wrapped "
    "in a Plan / Act / Observe / Reflect loop that enforces output constraints."
)

with st.sidebar:
    st.header("The goal")
    prompt = st.text_area("Story opening", "Once upon a time there was a dog", height=80)
    required = st.text_input("Must contain the word", "sandwich")
    forbidden = st.text_input("Must NOT contain (optional)", "")
    length = st.slider("Minimum words (0 = off)", 0, 150, 0, step=10)
    st.divider()
    retries = st.slider("Max attempts", 1, 8, 3)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)
    max_new = st.slider("Tokens to generate", 50, 250, 100, step=25)
    st.caption(
        "Generation is autoregressive on CPU: one forward pass per token, "
        "per attempt. Expect roughly a minute per attempt on a free tier."
    )
    go = st.button("Run the agent", type="primary", use_container_width=True)

with st.spinner("Loading the model (downloads ~115MB on first run)..."):
    model, enc, device, trained = load_model()

if not trained:
    st.error(
        "Running on an untrained model - the weights could not be downloaded. "
        f"The loop still works, but the prose will be gibberish. Source: {CHECKPOINT_URL}"
    )

if not go:
    st.info("Set a goal in the sidebar, then press **Run the agent**.")
    st.subheader("How it works")
    st.code(
        "PLAN     read the goal        -> prompt + constraint set\n"
        "ACT      sample a draft       -> SLM generates\n"
        "OBSERVE  validate the draft   -> deterministic string/logic checks\n"
        "REFLECT  on failure, change   -> inject a new prompt strategy, retry",
        language="text",
    )
    st.markdown(
        "The observer is **not** a second language model - it is plain string "
        "matching, so validation is deterministic, free and auditable."
    )
    st.stop()

constraints = [contains_word(required)] if required.strip() else []
if forbidden.strip():
    constraints.append(excludes_word(forbidden.strip()))
if length:
    constraints.append(min_words(length))

if not constraints:
    st.warning("Add at least one constraint.")
    st.stop()

agent = StoryDirectorAgent(model, enc, device=device, verbose=False)

# Persist across reruns: a plain button is only True on the run that fired it,
# so without this the whole trace disappears on the next widget interaction.
if go:
    with st.spinner(f"Running up to {retries} attempts on {device}..."):
        st.session_state["result"] = agent.run(
            prompt, constraints=constraints,
            max_retries=retries, max_new_tokens=max_new, temperature=temperature,
        )

if "result" not in st.session_state:
    st.info("Set a goal in the sidebar, then press **Run the agent**.")
    st.stop()

result = st.session_state["result"]

c1, c2, c3 = st.columns(3)
c1.metric("Outcome", "satisfied" if result.success else "gave up")
c2.metric("Attempts used", result.n_attempts)
c3.metric("Constraints", len(constraints))

st.subheader("The loop, attempt by attempt")
for a in result.attempts:
    icon = "PASS" if a.ok else "FAIL"
    with st.expander(f"Attempt {a.index} - {icon}", expanded=not a.ok or a.ok):
        st.markdown("**Prompt used**")
        st.code(a.prompt, language="text")
        st.markdown("**Draft**")
        st.write(a.text)
        if a.passed:
            st.success("Satisfied: " + ", ".join(a.passed))
        if a.failed:
            st.error("Failed: " + ", ".join(a.failed))
        if a.strategy:
            st.markdown("**REFLECT - strategy injected for the next attempt**")
            st.code(a.strategy, language="text")

st.subheader("Final story")
if result.success:
    st.success("All constraints satisfied.")
else:
    st.warning("Constraints were not satisfied within the attempt budget.")
st.write(result.text)
