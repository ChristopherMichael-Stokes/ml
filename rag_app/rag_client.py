from pathlib import Path

import gradio as gr
import llm_lib
import torch
from transformers import BitsAndBytesConfig

DEBUG = True
if DEBUG:
    base_path = Path("./rag_app")
else:
    base_path = Path("./")

# TODO: add config / env file

load_params = {
    # "quantization_config": BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # ),
    "torch_dtype": "auto",
    "device_map": "auto",
}

generate_params = {
    "temperature": 0.5,
    "max_new_tokens": 8192,
}

system_prompt = (
    "You are an expert career adviser and life coach helping users to plan their future."
    " Please respond truthfully and accurately, and if there is any CONTEXT information then use it to"
    " inform your answers.  Keep answers brief, do not repeat your self and do not include redundant information.",
)
llm = llm_lib.StreamingLLM(
    "hf",
    str(base_path / "qwen-2-7b-4bit/"),
    # str(base_path / "qwen-2-7b/"),
    # "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
    load_params,
    generate_params,
)  # , system_prompt=system_prompt)

# log.info("loading embedding")
embedder = llm_lib.construct_bge_model(str(base_path / "bge-large"))
# log.info("configuring app")


def echo(message, history):
    # TODO: implement some prompt token reduction techniques
    # ... or just limit the chat window to the make tokens length to preserve
    # fidelity

    # for thread in llm_lib.GENERATION_THREADS:
    #     # TODO: bug here with threads not getting closed properly when
    #     # 'stop is clicked, only terminating properly when the latest thread
    #     # is consumed
    #     print(thread)
    #     thread.join()  # Don't like this as could continue generating very long prompt
    # Best solution for now is to just disable stop button as it cannot be trusted !!!
    for thread in llm_lib.GENERATION_THREADS:
        thread._stp
        thread.interrupt()

    # TODO: fix the parsing of history thing, it's just an unlabeled list of all messages
    # from both sides:  Input should be a valid dictionary or instance of ChatMessage [type=model_type, input_value=['Can you give detailed a... with your job search!'], input_type=list]

    # history = llm_lib.ChatHistory.model_validate({"history": history})
    # llm.chat_history.history = history.history
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


demo = gr.ChatInterface(
    fn=echo,
    # examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}],
    title="Sophias Echo Bot",
    multimodal=True,
    stop_btn=gr.Button("Don't click, I'm broken 🥲🥺"),
).queue()


# log.info("starting app")
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)
# log.info("app up")
