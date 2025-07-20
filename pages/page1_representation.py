import gradio as gr
import os
import pandas as pd
from process.repo2data import extract_dataset
from process.count_ast import count_ast_on_files

def pipeline_idiom_representation(folder_path):
    out_dir = "files/dataset.pkl"

    if not os.path.exists(out_dir):
        # Step 1: run repo2data
        extract_dataset(folder_path, out_dir)
        # Step 2: run count_ast
        count_ast_on_files()
    
    try:
        df = pd.read_pickle(out_dir)
        preview = df.head()
        return f"Detected existing dataset file: {out_dir}\nPreview of the first few rows:\n{preview}"
    except Exception as e:
        return f"Failed to read existing dataset file: {e}"


def idiom_representation_ui():
    with gr.Column(visible=False) as section:
        gr.Markdown("## ✨ Step 1: Idiom Representation Identification")

        folder_path = gr.Textbox(label="Please enter the source code directory path", lines=1, value='../repomini/repotest') # ../repomini/repotest
        btn_run = gr.Button("Run Extraction and Analysis")
        output = gr.Textbox(lines=10, label="Result")

        btn_run.click(fn=pipeline_idiom_representation, inputs=folder_path, outputs=output)

    return section