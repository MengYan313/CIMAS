from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import sys
import re
from node_kinds import func_kind, block_kind, stmt_kind

model_path = "/nasdata/Model/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(model_path, local_files_only=True, device_map="auto")
model.eval()
print("Model has been automatically distributed to multiple GPUs")

# Get embedding (mean pooling)
def get_embedding(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True, max_length=8192)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu() # Mean pooling of all token outputs
    torch.cuda.empty_cache()
    return embedding

def get_pros_src_and_embedding(data):
    def parse_extent(extent):
        """Split 4 variables from the extent string"""
        match = re.match(r"(\d+)-(\d+)-(\d+)-(\d+)", extent)
        if match:
            start_line, start_column, end_line, end_column = map(int, match.groups())
            return start_line, start_column, end_line, end_column
        else:
            raise ValueError("Invalid extent format")

    def is_within_extent(extent_pre, extent_cur):
        """Determine whether extent_cur is within the range of extent_pre"""
        pre_start_line, pre_start_column, pre_end_line, pre_end_column = parse_extent(extent_pre)
        cur_start_line, cur_start_column, cur_end_line, cur_end_column = parse_extent(extent_cur)

        if (pre_start_line < cur_start_line or (pre_start_line == cur_start_line and pre_start_column <= cur_start_column)) and \
        (pre_end_line > cur_end_line or (pre_end_line == cur_end_line and pre_end_column >= cur_end_column)):
            return True
        return False

    projects = data['project']
    files = data['cppFile']
    asts = data['func_ast']
    srcs = data['func_src']
    pros_name, pros_src, pros_emb, pros_info = [], [], [], []
    
    for i, project_data in enumerate(asts):
        pro_name = projects[i]
        print("Processing project {}:{}".format(i, pro_name))
        pro_src, pro_emb, pro_info = [], [], []
        for j, file_data in enumerate(project_data):
            file_name = files[i][j]
            print("Processing file {}:{}".format(j, file_name))
            # if j == len(project_data) - 1:
            #     print("Processing file {}:{}".format(j, file_name))
            for k, func_data in enumerate(file_data):
                if len(func_data) < 10: # Filter code blocks with less than 10 nodes
                    continue
                print("Processing code block {}:{}".format(k, len(func_data)))
                extent_valid = "0-0-0-0"
                extent_root = ""
                for l, node_info in enumerate(func_data):
                    code_snippet = node_info.get("code_snippet", "")
                    kind = node_info.get("kind", "")
                    extent = node_info.get("extent", "")
                    ast_num = node_info.get("ast_num", "0")

                    if l == 0: # The extent of the first node is the range of the code segment
                        extent_root = extent

                    if code_snippet == None or code_snippet == "" or ast_num < 5:
                        continue

                    if kind in func_kind or kind in block_kind:
                        # print(f"{extent} func-{kind}: {code_snippet}") # repr(code_snippet)
                        # print("node_info", node_info)
                        pro_src.append(code_snippet)
                        embedding = get_embedding(code_snippet)
                        pro_emb.append(embedding)
                        pro_info.append([pro_name, file_name, extent_root, node_info]) # str str str dict
                    elif kind in stmt_kind:
                        if not is_within_extent(extent_valid, extent): # The current node cannot be within the range of the previous node
                            # print(f"{extent} stmt-{kind}: {code_snippet}")
                            # print("node_info", node_info)
                            extent_valid = extent # Update valid range
                            pro_src.append(code_snippet)
                            embedding = get_embedding(code_snippet)
                            pro_emb.append(embedding)
                            pro_info.append([pro_name, file_name, extent_root, node_info]) # str str str dict
                        else:
                            # print(f"xx {extent} stmt-{kind}: {code_snippet}")
                            pass
                

        print(len(pro_src), len(pro_emb), len(pro_info))
        sys.stdout.flush()
        if len(pro_src) >= 1000: # When the number of code nodes in the current project is greater than 1000, add it to the dataset
            pros_name.append(pro_name) # 1D list
            pros_src.append(pro_src) # 2D list
            pros_emb.append(pro_emb) # 2D list
            pros_info.append(pro_info) # 2D list

    return pros_name, pros_src, pros_emb, pros_info


def write_data(pros_name, pros_src, pros_emb, pros_info, output_path):
    print(len(pros_name), len(pros_src), len(pros_emb), len(pros_info))
    data = pd.DataFrame({'pros_name': pros_name, 'pros_src': pros_src, 'pros_emb': pros_emb, 'pros_info': pros_info})
    print(data.head())
    print("Columns:", data.columns, "Shape:", data.shape)
    pd.to_pickle(data, output_path)


def main():
    output_path = "../files/llama_embedding.pkl"
    data = pd.read_pickle('../files/dataset_ast.pkl')
    print(data.head())
    pros_name, pros_src, pros_emb, pros_info = get_pros_src_and_embedding(data)
    write_data(pros_name, pros_src, pros_emb, pros_info, output_path)

# nohup python3 llm_embedding.py > llm_embedding.log 2>&1 &
if __name__ == "__main__": # 993196
    main()