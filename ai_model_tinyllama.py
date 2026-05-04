# ai_model_tinyllama.py
import os
import time
import streamlit as st
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import pipeline, logging as hf_logging
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    hf_logging = None
    TRANSFORMERS_AVAILABLE = False

try:
    from huggingface_hub import login
except ImportError:
    login = None

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    load_dataset = None
    DATASETS_AVAILABLE = False

st.set_page_config(
    page_title="TinyLLaMA + Qwen Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)
if TRANSFORMERS_AVAILABLE:
    hf_logging.set_verbosity_error()

MODEL_DEFINITIONS = {
    "LLaMA 3.2-3B-Instruct": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Requires Hugging Face access and may need a token.",
        "dtype": torch.bfloat16 if TORCH_AVAILABLE else None,
    },
    "Qwen 2-0.5B-Instruct": {
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "description": "A smaller Qwen instruction model.",
        "dtype": None,
    },
    "DistilGPT2": {
        "model_id": "distilgpt2",
        "description": "A small CPU-friendly GPT-2 variant for quick testing without GPU.",
        "dtype": None,
    },
}

@st.cache_data(show_spinner=False)
def load_squad_dataframe():
    base_dir = os.path.join(os.path.dirname(__file__), "squad_csv")
    train_path = os.path.join(base_dir, "train-squad.csv")

    if os.path.exists(train_path):
        return pd.read_csv(train_path)

    if not DATASETS_AVAILABLE:
        st.error(
            "Local `squad_csv/train-squad.csv` not found and datasets library is not installed. "
            "Install datasets with `pip install datasets` or provide the local CSV file."
        )
        return pd.DataFrame()

    st.warning(
        "Local `squad_csv/train-squad.csv` not found. Falling back to Hugging Face SQuAD dataset download."
    )

    dataset = load_dataset("squad")
    return dataset["train"].to_pandas()

@st.cache_resource(show_spinner=False)
def initialize_pipeline(model_id: str, dtype=None):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "The transformers package is not installed. Install transformers to load models."
        )
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install torch to load transformer models, or run a CPU compatible setup."
        )

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    pipeline_kwargs = {
        "model": model_id,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if dtype is not None and device_map != "cpu":
        pipeline_kwargs["torch_dtype"] = dtype

    return pipeline("text-generation", **pipeline_kwargs)

def maybe_login(token: str) -> bool:
    if not token:
        return False
    if login is None:
        st.sidebar.warning("huggingface_hub is not installed. Install it to use token-based login.")
        return False
    try:
        login(token=token)
        return True
    except Exception as exc:
        st.sidebar.error(f"Hugging Face login failed: {exc}")
        return False

def generate_prompt(context: str, question: str) -> str:
    return (
        "Answer the question based on the passage.\n"
        f"Passage: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )

def extract_generated_text(output):
    if isinstance(output, list) and output:
        first_result = output[0]
        if isinstance(first_result, dict):
            return first_result.get("generated_text", str(first_result))
        return str(first_result)
    return str(output)

st.title("TinyLLaMA + Qwen Streamlit Demo")
st.markdown(
    "Use this demo to explore question answering with the SQuAD dataset and three transformer models: "
    "LLaMA 3.2-3B-Instruct, Qwen 2-0.5B-Instruct, and a smaller DistilGPT2 CPU-friendly option."
)

with st.sidebar:
    st.header("Configuration")
    if not TRANSFORMERS_AVAILABLE:
        st.warning(
            "Transformers is not installed. Install it with `pip install transformers` to enable model loading."
        )
    elif not TORCH_AVAILABLE:
        st.warning(
            "PyTorch is not installed. Model loading will fail until you install torch. "
            "Install it with `pip install torch` or use a Python environment that already includes it."
        )
    hf_token = st.text_input("Hugging Face token", type="password")
    if hf_token:
        st.info("Token entered. The app will attempt to use it for gated model access.")
    default_model = "DistilGPT2" if not TORCH_AVAILABLE or not torch.cuda.is_available() else list(MODEL_DEFINITIONS.keys())[0]
    selected_models = st.multiselect(
        "Select models",
        list(MODEL_DEFINITIONS.keys()),
        default=[default_model],
    )
    max_new_tokens = st.slider("Max new tokens", 16, 256, 64, step=8)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, step=0.05)
    input_mode = st.radio("Input mode", ["SQuAD sample", "Custom prompt"])
    use_sample = input_mode == "SQuAD sample"
    st.markdown("---")
    st.markdown(
        "If a model requires gated Hugging Face access, provide a valid token or set the environment variable "
        "`HUGGINGFACE_TOKEN` before running Streamlit."
    )

train_df = load_squad_dataframe()

if use_sample:
    row_id = st.number_input(
        "SQuAD train row", min_value=0, max_value=len(train_df) - 1, value=0, step=1
    )
    context = st.text_area("Context", train_df.loc[row_id, "context"], height=250)
    question = st.text_input("Question", train_df.loc[row_id, "question"])
    true_answer = train_df.loc[row_id, "text"] if "text" in train_df.columns else None
    st.markdown("**Reference answer**")
    st.write(true_answer)
else:
    context = st.text_area("Context", "Enter a passage here.", height=250)
    question = st.text_input("Question", "Enter a question about the passage.")
    true_answer = None

prompt = generate_prompt(context, question)

if st.button("Generate answers"):
    if not selected_models:
        st.warning("Select at least one model from the sidebar.")
    elif not context or not question:
        st.warning("Provide both a context passage and a question.")
    else:
        if hf_token:
            maybe_login(hf_token)
        elif os.getenv("HUGGINGFACE_TOKEN"):
            maybe_login(os.getenv("HUGGINGFACE_TOKEN"))

        for model_name in selected_models:
            model_info = MODEL_DEFINITIONS[model_name]
            st.subheader(model_name)
            st.write(model_info["description"])
            try:
                with st.spinner(f"Loading {model_name}..."):
                    model_pipeline = initialize_pipeline(model_info["model_id"], model_info["dtype"])
                with st.spinner(f"Generating answer with {model_name}..."):
                    start_time = time.time()
                    output = model_pipeline(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                    elapsed = time.time() - start_time
                generated_text = extract_generated_text(output)
                st.success(f"Generated in {elapsed:.2f} seconds")
                st.code(generated_text)
            except Exception as exc:
                st.error(f"Unable to run {model_name}: {exc}")

with st.expander("Batch evaluate the first 10 SQuAD examples"):
    if st.button("Run batch evaluation"):
        if not selected_models:
            st.warning("Select at least one model before batch evaluation.")
        else:
            batch_results = []
            if hf_token:
                maybe_login(hf_token)
            elif os.getenv("HUGGINGFACE_TOKEN"):
                maybe_login(os.getenv("HUGGINGFACE_TOKEN"))

            available_models = []
            for model_name in selected_models:
                model_info = MODEL_DEFINITIONS[model_name]
                try:
                    initialize_pipeline(model_info["model_id"], model_info["dtype"])
                    available_models.append(model_name)
                except Exception as exc:
                    st.error(f"Could not load {model_name}: {exc}")

            for i in range(10):
                row = train_df.loc[i]
                batch_prompt = generate_prompt(row["context"], row["question"])
                row_result = {
                    "row": i,
                    "question": row["question"],
                    "true_answer": str(row["text"]) if "text" in row else None,
                }
                for model_name in available_models:
                    model_info = MODEL_DEFINITIONS[model_name]
                    model_pipeline = initialize_pipeline(model_info["model_id"], model_info["dtype"])
                    start_time = time.time()
                    output = model_pipeline(batch_prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                    row_result[f"{model_name} answer"] = extract_generated_text(output)
                    row_result[f"{model_name} latency"] = f"{time.time() - start_time:.2f}s"
                batch_results.append(row_result)

            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", csv, "squad_batch_results.csv", "text/csv")

st.markdown("---")
st.write(
    "This app loads a sample of the SQuAD dataset and lets you compare generation from different "
    "Hugging Face models. If you do not have access to LLaMA or Qwen models, use a different model ID "
    "or provide a valid Hugging Face token."
)
