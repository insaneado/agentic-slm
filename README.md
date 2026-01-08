# Agentic SLM: The Story Director

## Overview
This project demonstrates the implementation of a Small Language Model (SLM) built from scratch using PyTorch and an Agentic Workflow that acts as a "Director" to control the model's output.

While Large Language Models (LLMs) are powerful, they are resource-intensive. This project investigates how Rule-Based Agents can guide Small Models (approximately 15M parameters) to achieve complex instruction following that they cannot achieve on their own.

## The Architecture (The Brain)
The model is a Decoder-only Transformer (GPT-style) trained on the TinyStories dataset.

- **Architecture:** Custom PyTorch implementation of Self-Attention, Feed-Forward Networks, and LayerNorm.
- **Parameters:** ~15 Million (Small Language Model).
- **Context Length:** 128 tokens.
- **Training Data:** TinyStories (A synthetic dataset of short stories for children).
- **Tokenizer:** GPT-2 Byte-Pair Encoding (BPE).

## The Agent (The Director)
The core innovation of this project is the StoryDirectorAgent. A raw SLM is often stubborn and hallucinates. The Agent wraps the model in a reasoning loop to enforce user constraints.

### How it Works (Neuro-Symbolic Loop)
1. **PLAN:** The Agent accepts a goal (e.g., "Write a story about a cat that includes a sandwich").
2. **ACT:** The SLM generates a draft.
3. **OBSERVE:** The Agent scans the output using string matching and logic checks.
4. **REFLECT & CORRECT:**
   - If the constraint is met, the process concludes successfully.
   - If failed, the Agent dynamically injects a new prompt strategy (e.g., "Suddenly, a huge sandwich appeared...") and forces a regeneration.

This transforms a non-deterministic probabilistic model into a reliable system.

## Installation & Usage

### 1. Clone the Repository
git clone https://github.com/your-username/agentic-slm-project.git
cd agentic-slm-project

### 2. Install Dependencies
pip install torch numpy datasets tiktoken tqdm requests

### 3. Run the Project
Open the Jupyter Notebook `Final_Agentic_SLM.ipynb` and run all cells.
- **Training:** The notebook contains the full training loop.
- **Inference:** The final cell launches the Interactive Agent.

## Results

**Scenario:** The user asks for a story about a *dog* that must contain a *sandwich*.

**Attempt 1 (Raw Model Failure):**
> *Draft:* "Once upon a time there was a dog. He liked to run in the park. He saw a cat..."
> *Agent Decision:* FAIL. Word 'sandwich' missing.

**Attempt 2 (Agent Intervention):**
> *Agent Action:* Injecting Strategy B: "...Suddenly, a huge sandwich appeared."
> *Draft:* "Once upon a time there was a dog... Suddenly, a huge sandwich appeared. The dog was so happy and ate it."
> *Agent Decision:* SUCCESS.

## Tech Stack
- **Core:** Python, PyTorch (nn.Module, functional API).
- **Data Processing:** HuggingFace Datasets, TikToken.
- **Visualization:** Matplotlib.

## Future Improvements
- Scale the model to 100M parameters for better coherence.
- Implement a Critic LLM instead of regex-based checking for more nuance.
- Add Chain of Thought data to the training set.

## Credits
- Architecture based on "Attention Is All You Need" and Andrej Karpathy's nanoGPT.
- Dataset by Microsoft Research (TinyStories).
