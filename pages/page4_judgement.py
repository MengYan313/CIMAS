import gradio as gr
import pandas as pd
from judgement.local_with_local_filter import run_judgement_and_synthesis
import os

def pipeline_judgement(file_input, path_input, key, model, base_url, test_mode_radio):
    # Parse test mode
    test_mode = test_mode_radio == "open"
    # Prefer to use the path entered in the textbox, if empty then use the uploaded file
    if path_input and os.path.exists(path_input):
        file_path = path_input
    elif file_input is not None:
        file_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
    else:
        return "Please upload a file or enter a valid file path!"
    
    output_path = 'files/telephony_core_service.pkl'

    if not os.path.exists(output_path):
        run_judgement_and_synthesis(file_path, key, model, base_url, test_mode)
    
    # Read pkl file content and output the first four columns to the table
    try:
        df = pd.read_pickle(output_path)
        # If df is a list, convert to DataFrame
        if isinstance(df, list):
            df = pd.DataFrame(df)
        # Only take the first four columns
        df_preview = df.iloc[:, :4]
        return f"✅ Judgment and synthesis completed.\nResult file saved at: {output_path}", df_preview
    except Exception as e:
        return f"✅ Judgment and synthesis completed, but failed to read result file: {e}", None


def judgement_ui():
    with gr.Column(visible=False) as section:
        gr.Markdown("## 🔍 Step 4: Idiom Judgment and Synthesis")

        file_input = gr.File(label="Select the output pkl file from Step 3", file_types=[".pkl"], file_count="single")
        path_input = gr.Textbox(label="Or directly enter the output pkl file path from Step 3", lines=1, value='files/clusters_with_abstracted.pkl') # files/clusters_with_abstracted.pkl
        key_input = gr.Textbox(label="API Key", lines=1, value='sk-Amr2bl7ce65IyzQuK3OkACqPDXeg5xF9OrMDIOOQVOqKWOh8', type="password") # sk-Amr2bl7ce65IyzQuK3OkACqPDXeg5xF9OrMDIOOQVOqKWOh8
        model_input = gr.Textbox(label="Model", lines=1, value='gpt-4o') # gpt-4o
        base_url_input = gr.Textbox(label="Base URL", lines=1, value='https://us.ifopen.ai/v1') # https://us.ifopen.ai/v1
        test_mode_radio = gr.Radio(choices=["close", "open"], value="close", label="Test Mode")
        btn_run = gr.Button("Run Judgment & Synthesis")
        output_text = gr.Textbox(lines=5, label="Result")
        output_table = gr.Dataframe(label="Idiom Display Table", wrap=True)

        btn_run.click(
            fn=pipeline_judgement,
            inputs=[file_input, path_input, key_input, model_input, base_url_input, test_mode_radio],
            outputs=[output_text, output_table]
        )

    return section