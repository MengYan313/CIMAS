import gradio as gr
import pandas as pd
from abstract.abstraction import run_abstraction_pipeline
import os

def pipeline_abstraction(pkl_file, pkl_path_text):
    # Prefer to use the path entered in the textbox, if empty then use the uploaded file
    if pkl_path_text and os.path.exists(pkl_path_text):
        file_path = pkl_path_text
    elif pkl_file is not None:
        file_path = pkl_file.name if hasattr(pkl_file, 'name') else str(pkl_file)
    else:
        return "Please upload a file or enter a valid file path!"

    output_path = file_path[:-4] + '_with_abstracted.pkl'
    csv_path = 'files/abstract_compare.csv'

    if not os.path.exists(output_path) or not os.path.exists(csv_path):
        run_abstraction_pipeline(file_path)
    
    try:
        # Read the csv file and output the first 100 rows
        df = pd.read_csv(csv_path)
        preview = df.head(100)
        text = f"✅ Abstraction completed, result saved at: {output_path}\nPreview of the first 100 rows of abstraction comparison below:"
        return text, preview
    except Exception as e:
        return f"Abstraction completed, but failed to read result file: {e}", None


def abstraction_ui():
    with gr.Column(visible=False) as section:
        gr.Markdown("## 🧠 Step 3: Differentiating Element Abstraction")

        file_input = gr.File(label="Select the output pkl clustering result file from Step 2", file_types=[".pkl"], file_count="single")
        path_input = gr.Textbox(label="Or directly enter the output pkl file path from Step 2", lines=1, value='files/clusters.pkl') # files/clusters.pkl
        btn_run = gr.Button("Run Abstraction Process")
        output_text = gr.Textbox(lines=5, label="Result")
        output_table = gr.Dataframe(label="Abstraction Result Preview", wrap=True)

        btn_run.click(fn=pipeline_abstraction, inputs=[file_input, path_input], outputs=[output_text, output_table])

    return section