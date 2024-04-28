#!/bin/bash

# /usr/bin/python3.10 rag_client.py 2>&1 | tee output/output.txt
# /usr/bin/python3.10 gradio_demo.py 2>&1 | tee output/output.txt
# uvicorn --host 0.0.0.0 --port 7860 gradio_demo:app 2>&1 | tee output/output.txt
/usr/bin/python3.10 -m uvicorn --workers 1 --host 0.0.0.0 --port 7860 rag_client:app 2>&1 | tee output/output.txt
