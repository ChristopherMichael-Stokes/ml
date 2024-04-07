# Playground local rag app based on gradio + huggingface / llama cpp

## TODOs

### code improvements

- [ ]: unit tests
- [ ]: OmegaConf
- [ ]: framework agnostic llm inference streaming
- [ ]: fix the streaming issue with huggingface models - probably means a refactor from threads to multiprocessing + object sharing (definitely cannot afford to instanciate llm in a new process every time we do inference)

### feature improvements

- [ ]: functionality to export and download current chat to docx
- [ ]: better rag support
