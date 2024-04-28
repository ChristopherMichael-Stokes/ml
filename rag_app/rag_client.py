import logging
from pathlib import Path

import gradio as gr
import llm_lib
from fastapi import FastAPI
from omegaconf import DictConfig, ListConfig, OmegaConf

# Create a logger object
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Create a file handler which logs even debug messages
fh = logging.FileHandler("output/info.log", mode="w")
fh.setLevel(logging.DEBUG)


# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

# Add handlers to the logger
log.addHandler(fh)

app = FastAPI()


log.info("Reading config")
base_conf = OmegaConf.load(Path(__file__).parent / "config.yaml")
cli_conf = OmegaConf.from_cli()
conf = OmegaConf.merge(base_conf, cli_conf)

load_params = conf.load_params if "load_params" in conf else {}
generate_params = conf.generation_params if "generation_params" in conf else {}
system_prompt = conf.system_prompt if "system_prompt" in conf else None

log.info("Loading llm")
llm = llm_lib.StreamingLLM(
    conf.model_backend, conf.llm_model, load_params, generate_params, system_prompt
)

log.info("Loading embedding model")
embedder = llm_lib.construct_bge_model(conf.embed_model)


def echo(message, history):
    # TODO: implement some prompt token reduction techniques
    # ... or just limit the chat window to the make tokens length to preserve
    # fidelity
    llm.chat_history.history = []
    for i, content in enumerate(history):
        llm.chat_history.history.append(
            llm_lib.ChatMessage.model_validate({"role": "user", "content": content[0]})
        )
        if not content[1]:
            break
        llm.chat_history.history.append(
            llm_lib.ChatMessage.model_validate({"role": "assistant", "content": content[1]})
        )

    prompt = message["text"]
    if "files" in message and message["files"]:
        # update the rag index
        vector_store = llm_lib.construct_rag_store([f["path"] for f in message["files"]], embedder)
        context = llm_lib.context_retrieval({"prompt": prompt}, vector_store)
        prompt = f"{prompt}\n\nCONTEXT\n{context}"

    completion = llm.generate(prompt)

    next(completion)  # skip assistant line
    for part in completion:
        yield part


log.info("Building ui")

demo = gr.ChatInterface(
    fn=echo,
    title="Sophias Echo Bot",
    multimodal=True,
    stop_btn=gr.Button("Don't click, I'm broken 🥲🥺") if conf.model_backend == "hf" else "Stop",
).queue()

log.info("Launching server")
app = gr.mount_gradio_app(app, demo, path="/")
