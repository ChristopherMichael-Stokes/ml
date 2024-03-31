import logging

import gradio as gr
import llm_lib

logging.basicConfig(filename="output/info.log", level=logging.INFO)

log = logging.getLogger(__name__)

# TODO: add config / env file

log.info("Loading llm")
# llm = llm_lib.construct_ggml_model("qwen-2-7b/ggml-model-Q4_K_M.gguf", stream=True) #context_size=8192, stream=True)
llm = llm_lib.construct_ggml_model("qwen-2-7b/ggml-model-Q4_K_M.gguf", context_size=32768, stream=True)
log.info("loading embedding")
embedder = llm_lib.construct_bge_model("bge-large")
log.info("configuring app")


def echo(message, history):

    # TODO: Do some proper conversation chaining

    prompt = message["text"]
    prompt_message = [
        {
            "role": "system",
            "content": "You are an expert career adviser and life coach helping users to plan their future."
            " Please respond truthfully and accurately, and if there is any CONTEXT information then use it to"
            " inform your answers.  Keep answers brief, do not repeat your self and do not include redundant information.",
        },
    ]
    if "files" in message and message["files"]:
        # update the rag index
        vector_store = llm_lib.construct_rag_store([f["path"] for f in message["files"]], embedder)

        context = llm_lib.context_retrieval({"prompt": prompt}, vector_store)
        prompt_message.append({"role": "user", "content": f"{prompt}\n\nCONTEXT\n{context}"})
    else:
        # just prompt without context
        prompt_message.append({"role": "user", "content": prompt})

    completion = llm.create_chat_completion(messages=prompt_message, temperature=0.1, stream=True)

    next(completion)  # skip assistant line

    text = ""
    for tok in completion:
        if not tok["choices"]:
            break

        choice = tok["choices"][0]
        if not choice["delta"]:
            break

        text += choice["delta"]["content"]
        yield text
    # return completion["choices"][0]["message"]["content"]


demo = gr.ChatInterface(
    fn=echo,
    # examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}],
    title="Sophias Echo Bot",
    multimodal=True,
).queue()


log.info("starting app")
demo.launch(server_name='0.0.0.0', server_port=7860, )
log.info("app up")
