import gradio as gr
import pandas as pd
import os
from mining.code_embedding import generate_embeddings
from mining.clustering import run_clustering

def pipeline_idiom_mining(pkl_file, pkl_path_text):
    # Prefer to use the path entered in the textbox, if empty then use the uploaded file
    if pkl_path_text and os.path.exists(pkl_path_text):
        file_path = pkl_path_text
    elif pkl_file is not None:
        file_path = pkl_file.name if hasattr(pkl_file, 'name') else str(pkl_file)
    else:
        return "Please upload a file or enter a valid file path!"

    output_dir = "files/"
    embedding_file = os.path.join(output_dir, "embedding.pkl")
    cluster_result_path = os.path.join(output_dir, "clusters.pkl")
    csv_path = os.path.join(output_dir, "clusters_top100.csv")

    # Check if clustering results exist
    if not os.path.exists(cluster_result_path) or not os.path.exists(csv_path):
        # 1️⃣ Generate code embeddings
        generate_embeddings(file_path, embedding_file)
        # 2️⃣ Clustering analysis
        run_clustering(embedding_file, cluster_result_path)
        return f"✅ Code embedding and clustering analysis completed.\n\nInput file: {file_path}\nEmbedding file: {embedding_file}\nClustering result: {cluster_result_path}", None

    try:
        df = pd.read_csv(csv_path, skiprows=3, on_bad_lines='skip')
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                second_line = lines[1].strip().split(",")
                cluster_num = second_line[0] if len(second_line) > 0 else "Unknown"
                avg_cluster_size = second_line[1] if len(second_line) > 1 else "Unknown"
            else:
                cluster_num = "Unknown"
                avg_cluster_size = "Unknown"
        text = (
            f"Detected existing clustering result: {cluster_result_path}\n"
            f"clusters_top100.csv loaded.\n"
            f"Number of clusters: {cluster_num}\nAverage cluster size: {avg_cluster_size}"
        )
        return text, df
    except Exception as e:
        return f"Failed to read csv file: {e}", None


def idiom_mining_ui():
    with gr.Column(visible=False) as section:
        gr.Markdown("## 🧩 Step 2: Idiomatic Code Mining")

        file_input = gr.File(label="Select the output pkl file from Step 1", file_types=[".pkl"], file_count="single")
        path_input = gr.Textbox(label="Or directly enter the output pkl file path from Step 1", lines=1, value='files/dataset_ast.pkl') # files/dataset_ast.pkl
        btn_run = gr.Button("Run Mining Process")
        output_text = gr.Textbox(lines=10, label="Result")
        output_table = gr.Dataframe(label="Clustering Result Table", wrap=True)

        btn_run.click(
            fn=pipeline_idiom_mining,
            inputs=[file_input, path_input],
            outputs=[output_text, output_table]
        )

    return section