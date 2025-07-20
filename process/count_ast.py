import pandas as pd
import sys
from dataclasses import dataclass
from .node_kinds import func_kind, block_kind, stmt_kind
import random

@dataclass
class LocLabelInfo:
    pro_name: str      # Project name
    loc_label: str     # Code location label
    is_train: bool     # Whether it belongs to the training set
    total_num: int     # Total number of nodes (len(func))
    idiom_num: int     # Number of nodes that are idioms

def count_children(func_ast, index):
    current_depth = func_ast[index]['depth']
    child_count = 0
    for i in range(index + 1, len(func_ast)):
        if func_ast[i]['depth'] > current_depth:
            child_count += 1
        elif func_ast[i]['depth'] <= current_depth:
            break
    return child_count + 1  # The node itself is also counted as one

def add_child_num_to_func_ast(data):
    projects = data['project']
    files = data['cppFile']
    asts = data['func_ast']
    loc_info_list = []  # Store all function information

    for i, project in enumerate(asts):
        pro_name = projects[i]
        print("Currently processing project {}:{}".format(i, pro_name))
        for j, cpp_file in enumerate(project):
            file_name = files[i][j]
            print("Currently processing file {}:{}".format(j, file_name))
            for k, func in enumerate(cpp_file):
                if len(func) < 10:  # Filter code blocks with less than 10 nodes in code_embedding.py
                    continue
                extent = func[0].get("extent", "")  # Code segment location
                loc_label = f"{pro_name}-{file_name}-{extent}"
                valid_num = 1
                
                for l, node_info in enumerate(func):
                    node_info['ast_num'] = count_children(func, l)
                    kind = node_info.get("kind", "")
                    if kind in func_kind or kind in block_kind or kind in stmt_kind:
                        valid_num += 1
                
                loc_info_list.append(LocLabelInfo(
                    pro_name=pro_name,
                    loc_label=loc_label,
                    is_train=False,  # Default to False, update when splitting the training set later
                    total_num=len(func),
                    idiom_num=0  # idiom_num is currently not used, set to 0
                ))
                print(f"{loc_label} : {len(func)} -> {valid_num}")
                sys.stdout.flush()
    
    return data, loc_info_list  # Return the function information list

def split_loc_labels(func_info_list):
    """ Split into training and test sets by project """
    # Group loc_label by project
    projects = {}
    for func_info in func_info_list:
        project_name = func_info.pro_name
        if project_name not in projects:
            projects[project_name] = []
        projects[project_name].append(func_info)
    
    # For each project's loc_label, split 7:3
    for project_name, func_infos in projects.items():
        random.shuffle(func_infos)  # Shuffle randomly
        split_index = int(len(func_infos) * 0.7)
        for func_info in func_infos[:split_index]:
            func_info.is_train = True  # Mark as training set
        for func_info in func_infos[split_index:]:
            func_info.is_train = False  # Mark as test set

def main():
    input_file_path = '../files/dataset.pkl'
    output_file_path = '../files/dataset_ast.pkl'
    loc_info_file_path = '../files/loc_info.pkl'
    
    data = pd.read_pickle(input_file_path) # Read data
    updated_data, loc_info_list = add_child_num_to_func_ast(data) # Add child_num key-value pair and generate function info list
    split_loc_labels(loc_info_list) # Split into training and test sets and save
    
    pd.to_pickle(updated_data, output_file_path) # Save data
    print(f"Data has been saved to {output_file_path}")
    pd.to_pickle(loc_info_list, loc_info_file_path) # Save loclabel info list
    print(f"loclabel info has been saved to {loc_info_file_path}")


def count_ast_on_files():
    input_file_path = 'files/dataset.pkl'
    output_file_path = 'files/dataset_ast.pkl'
    loc_info_file_path = 'files/loc_info.pkl'
    
    data = pd.read_pickle(input_file_path) # Read data
    updated_data, loc_info_list = add_child_num_to_func_ast(data) # Add child_num key-value pair and generate function info list
    split_loc_labels(loc_info_list) # Split into training and test sets and save
    
    pd.to_pickle(updated_data, output_file_path) # Save data
    print(f"Data has been saved to {output_file_path}")
    pd.to_pickle(loc_info_list, loc_info_file_path) # Save loclabel info list
    print(f"loclabel info has been saved to {loc_info_file_path}")


# nohup python3 count_ast.py > count_ast.log 2>&1 &
# nohup python3 -m data_process.count_ast > count_ast.log 2>&1 &
if __name__ == "__main__":
    main()