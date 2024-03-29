from functools import partial
from pathlib import Path
from typing import List

import gradio as gr
import llm_lib
import torch
from langchain_community.vectorstores.faiss import FAISS

chain = None
streamer = None


nf4_config = llm_lib.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
llm, streamer = llm_lib.construct_gen_model("google/gemma-2b-it", nf4_config)
embedder = llm_lib.construct_bge_model()
pdfs = list(Path("notebooks/data").absolute().iterdir())
# vector_store = llm_lib.construct_rag_store(pdfs, embedder)

vector_store = FAISS.from_texts([""], embedder)


prompt = llm_lib.ChatPromptTemplate.from_template(
    "<bos><start_of_turn>user\n"
    # "Using your own internal knowledge combined with the following contextual data (denoted by CONTEXT), "
    # "please answer my prompt (denoted by PROMPT)\nCONTEXT\n{context}\nPROMPT\n"
    "{context}\n\n"
    "{prompt}<end_of_turn>\n<start_of_turn>model\n"
)

chain = llm_lib.construct_chain(prompt=prompt, llm=llm, vector_store=vector_store)


def reindex(pdfs: List[Path], *args, **kwargs):
    global chain, vector_store, embedder
    vector_store = llm_lib.construct_rag_store(pdfs, embedder)
    chain = llm_lib.construct_chain(prompt=prompt, llm=llm, vector_store=vector_store)


with gr.Blocks() as demo:
    gr.Markdown("## Sophia's __Chatter__ bot")
    with gr.Column():
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(
                    label="Chat Message Box", placeholder="Chat Message Box", show_label=False
                )
                upload = gr.UploadButton(file_count="multiple", file_types=["pdf"])
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

    def respond(message, chat_history):
        converted_chat_history = ""
        if len(chat_history) > 0:
            # TODO: Handle the proper prompt format
            for c in chat_history[-2:]:
                converted_chat_history += (
                    f"<|prompter|>{c[0]}<|endoftext|><|assistant|>{c[1]}<|endoftext|>"
                )

        prompt = f"{converted_chat_history}<|prompter|>{message}<|endoftext|><|assistant|>"

        # send request to endpoint
        llm_response = chain.invoke({"prompt": prompt})

        # remove prompt from response
        parsed_response = llm_response.split("<start_of_turn>model\n")[-1]  # Handle this ending thing
        chat_history.append((message, parsed_response))
        return "", chat_history

    submit.click(respond, [message, chatbot], [message, chatbot], show_progress="full", queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    upload.click(reindex)  # TODO: Handle the file upload and insertion into vector db

demo.launch(debug=True)
