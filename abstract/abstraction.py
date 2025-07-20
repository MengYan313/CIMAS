import pandas as pd
import re
from collections import Counter
import json
import random

def tokenize_code(code):
    """
    Split code snippet into a sequence of tokens.
    - Strings inside quotes are treated as a single token.
    - Other parts are tokenized by the original rules.
    """
    # Match strings inside single or double quotes
    pattern = r'(\".*?\"|\'.*?\'|\w+|[^\s\w])'
    return re.findall(pattern, code)


def get_token_spans(code, tokens):
    """Return the start and end positions of each token in the original string."""
    spans = []
    idx = 0
    for token in tokens:
        # Skip whitespace
        while idx < len(code) and code[idx].isspace():
            idx += 1
        start = idx
        end = idx + len(token)
        spans.append((start, end))
        idx = end
    return spans

def join_tokens_like_original(code, tokens):
    """
    Concatenate abstracted tokens according to the original code's token spacing.
    If there is no whitespace between original tokens, no space is added after abstraction;
    if there is any whitespace between original tokens, add a single space after abstraction.
    """
    spans = get_token_spans(code, tokenize_code(code))
    # Since the length of abstracted tokens may differ from the original, here we assume a one-to-one correspondence
    merged = ''
    for i, token in enumerate(tokens):
        merged += token
        if i < len(tokens) - 1:
            # Get the gap between original tokens
            _, end = spans[i]
            next_start, _ = spans[i + 1]
            gap = code[end:next_start]
            if any(c.isspace() for c in gap):
                merged += ' '
            # Otherwise, do not add space
    return merged


def guess_token_type(token, tokens, idx):
    # Common type names (expand as needed)
    type_keywords = {
        "void", "char", "short", "int", "long", "float", "double", "bool",
        "size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "wchar_t", "string"
    }
    # Check if it's a type name
    if token in type_keywords:
        return "<TYPE>"
    # Check if it's a function name
    if idx + 1 < len(tokens) and tokens[idx + 1] == '(':
        return "<FUN>"
    # Check if it's a type name (C/C++ style, capitalized or preceded by const/static/struct/class/typedef etc.)
    if (token[0].isupper() and token.isidentifier()) or \
       (idx > 0 and tokens[idx - 1] in {"const", "struct", "class", "typedef", "enum"}):
        return "<TYPE>"
    # Otherwise, treat as variable name
    return "<VAR>"

def abstract_frequent_idiom(frequent_idiom, normal_idioms, threshold=0.8):
    frequent_tokens = tokenize_code(frequent_idiom)
    normal_tokens_list = [tokenize_code(code) for code in normal_idioms]
    max_length = len(frequent_tokens)
    normal_tokens_list = [tokens[:max_length] for tokens in normal_tokens_list]

    # Sliding window approach
    token_marks = [False] * len(frequent_tokens)
    for i in range(len(frequent_tokens) - 1):
        token_pair = (frequent_tokens[i], frequent_tokens[i + 1])
        match_count = 0
        for normal_tokens in normal_tokens_list:
            for j in range(len(normal_tokens) - 1):
                if normal_tokens[j] == token_pair[0] and normal_tokens[j + 1] == token_pair[1]:
                    match_count += 1
                    break
        match_ratio = match_count / len(normal_tokens_list)
        if match_ratio >= threshold:
            token_marks[i] = True
            token_marks[i + 1] = True

    # First token check
    first_token = frequent_tokens[0]
    first_count = sum(tokens[0] == first_token for tokens in normal_tokens_list if tokens)
    if first_count / len(normal_tokens_list) >= threshold:
        token_marks[0] = True

    # Last token check
    last_token = frequent_tokens[-1]
    last_count = sum(tokens[-1] == last_token for tokens in normal_tokens_list if tokens)
    if last_count / len(normal_tokens_list) >= threshold:
        token_marks[-1] = True

    abstracted_tokens = []
    for i, token in enumerate(frequent_tokens):
        if token_marks[i]:
            abstracted_tokens.append(token)
        else:
            if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
                abstracted_tokens.append("<STR>")
            elif token.isdigit():
                abstracted_tokens.append("<CONST>")
            elif re.match(r'^[a-zA-Z_]\w*$', token):
                # Further distinguish variable/function/type
                token_type = guess_token_type(token, frequent_tokens, i)  
                if token_type == "<FUN>":
                    abstracted_tokens.append(token)  # Keep original function name
                else:
                    abstracted_tokens.append(token_type)
            else:
                # abstracted_tokens.append("<SYMBOL>")
                abstracted_tokens.append(token)  # Keep original token, no longer abstract as <SYMBOL>

    return join_tokens_like_original(frequent_idiom, abstracted_tokens)


def process_project_clusters(cluster_results_file, project_name, threshold=0.8):
    """
    Read the clustering result file and abstract frequent idioms for the specified project.
    :param cluster_results_file: Path to the clustering result file.
    :param project_name: Project name.
    :param threshold: Consistency threshold, default is 0.8 (80%).
    """
    # Read clustering result file
    cluster_results = pd.read_pickle(cluster_results_file)

    # Filter out the clustering data for the specified project
    project_data = next((item for item in cluster_results if item["pros_name"] == project_name), None)
    if not project_data:
        print(f"Project {project_name} not found in clustering data.")
        return

    clusters = project_data["clusters"]

    # Abstract frequent idioms for each cluster
    for _, row in clusters.iterrows():
        cluster_size = row["cluster_size"]
        center_point = row["center_point"]
        else_point = row["else_point"]

        if cluster_size > 1 and center_point and else_point:
            print(f"Processing cluster, size: {cluster_size}")
            
            # Print frequent idiom
            print("Frequent idiom:")
            print(center_point)
            
            # Print all normal idioms
            print("Normal idioms:")
            for idx, code in enumerate(else_point, start=1):
                print(f"Normal idiom {idx}:")
                print(code)
            
            # Abstract frequent idiom
            abstracted_idiom = abstract_frequent_idiom(center_point, else_point, threshold)
            
            # Check if abstraction succeeded (changed after removing whitespace)
            if "".join(abstracted_idiom.split()) != "".join(center_point.split()):
                print("*" * 40)
                print("Abstracted frequent idiom:")
                print(abstracted_idiom)
                print("*" * 39)
            else:
                print("Abstracted frequent idiom:")
                print(abstracted_idiom)
            print("-" * 80)


def process_all_projects_clusters(cluster_results_file, threshold=0.8):
    """
    Read the clustering result file, abstract frequent idioms for all projects, and store the results in a new column.
    """
    cluster_results = pd.read_pickle(cluster_results_file)

    for project_data in cluster_results:
        project_name = project_data.get("pros_name", "Unknown")
        clusters = project_data["clusters"]
        print("=" * 80)
        print(f"Project: {project_name}")
        print("=" * 80)
        total_count = 0
        abstracted_count = 0
        abstracted_list = []
        pre_list = []
        post_list = []
    
        for _, row in clusters.iterrows():
            cluster_size = row["cluster_size"]
            center_point = row["center_point"]
            else_point = row["else_point"]

            if cluster_size > 1 and center_point and else_point:
                total_count += 1
                print(f"Processing cluster, size: {cluster_size}", "Normal idiom count:", len(else_point), "Frequent idiom:")
                print(center_point)
                abstracted_idiom = abstract_frequent_idiom(center_point, else_point, threshold)
                if any("".join(center_point.split()) != "".join(code.split()) for code in else_point):
                    if "".join(abstracted_idiom.split()) != "".join(center_point.split()): # 1. Original code differs and abstraction succeeded
                        abstracted_count += 1
                        abstracted_list.append(abstracted_idiom)
                        pre_list.append(center_point)
                        post_list.append(abstracted_idiom)
                    else: # 2. Original code differs but abstraction failed
                        abstracted_list.append("same")
                else: # 3. Original code is identical so abstraction must fail
                    abstracted_list.append("same")
            else:
                abstracted_list.append("same")
        
        clusters["abstracted_center_point"] = abstracted_list
        print(f"Project {project_name} successfully abstracted frequent idioms: {total_count} -> {abstracted_count}")
        print("=" * 80)

        # Only keep successful abstraction comparisons
        compare_df = pd.DataFrame({
            "origin_center_point": pre_list,
            "abstracted_center_point": post_list
        })
        if not compare_df.empty:
            csv_file = 'files/abstract_compare.csv'
            compare_df.to_csv(csv_file, index=False)
            print(f"Abstract comparison csv file saved: {csv_file}")
        else:
            print(f"Project {project_name} has no successful abstraction comparison, no csv output.")

    # Save new pkl file with abstraction results
    output_file = cluster_results_file[:-4] + '_with_abstracted.pkl'
    pd.to_pickle(cluster_results, output_file)
    print(f"New pkl file with abstraction results saved: {output_file}")
    print("File type:", type(cluster_results), "Project count:", len(cluster_results), "First project keys:", cluster_results[0].keys())
    clusters = cluster_results[0]['clusters']
    print("DataFrame columns:", clusters.columns)
    print("First few rows:")
    print(clusters.head())


def test_abstract_frequent_idiom(threshold=0.8):
    """
    Test abstraction logic, input a frequent idiom and a list of normal idioms, output the abstraction result.
    """
    frequent_idiom = "validate_and_test_gpu_preference(display_name, true)"
    normal_idioms = [
        "validate_and_test_gpu_preference(display_name, false)"
    ]
    print("Frequent idiom:")
    print(frequent_idiom)
    frequent_tokens = tokenize_code(frequent_idiom)
    print("Frequent idiom token list:")
    print(frequent_tokens)
    print("Normal idioms:")
    for idx, code in enumerate(normal_idioms, start=1):
        print(f"Normal idiom {idx}:")
        print(code)
        print(f"Normal idiom {idx} token list:")
        print(tokenize_code(code))
    abstracted_idiom = abstract_frequent_idiom(frequent_idiom, normal_idioms, threshold)
    print("Abstracted token list:")
    print(tokenize_code(abstracted_idiom))
    if "".join(abstracted_idiom.split()) != "".join(frequent_idiom.split()):
        print("*" * 40)
        print("Abstracted frequent idiom:")
        print(abstracted_idiom)
        print("*" * 39)
    else:
        print("Abstracted frequent idiom:")
        print(abstracted_idiom)
    print("-" * 80)


def run_abstraction_pipeline(file_path):
    process_all_projects_clusters(file_path, threshold=0.8)


# nohup python3 abstraction.py > abstraction.log 2>&1 &
if __name__ == "__main__": # 1819094
    # test_abstract_frequent_idiom()
    cluster_results_file = "../files/gcb_cluster_results.pkl"
    # process_project_clusters(cluster_results_file, "TrafficMonitor")