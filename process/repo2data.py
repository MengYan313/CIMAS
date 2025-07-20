import clang.cindex
from clang.cindex import CursorKind
import os
import re
import pandas as pd
from typing import List
import argparse

class GetCppFile:
    '''This class is used to extract all .cpp files from a given directory'''
    def __init__(self, projects: List[str], pro_file_list: List[str]) -> None:
        self.projects = projects  # Store all project names, i.e., projectStr
        self.pro_file_list = pro_file_list  # Store all .cpp file paths, 2D list

    def get_cpp_projects(self, PathStr: str):  # --PathStr = /home/user/repository
        # os.walk() returns a generator that recursively traverses all directories and files under the specified path
        # root: string of the current directory path, representing the path of the currently traversed directory
        # dirs: list of all subdirectories in the current directory (excluding files), empty list if none
        # files: list of all files in the current directory (excluding subdirectories), empty list if none
        for root, dirs, files in os.walk(PathStr):
            self.projects = dirs # Only keep directories, i.e., projectStr
            self.projects.sort()  # Sort by name
            break # Stop when traversing the first layer of PathStr, so only return the first layer of subdirectories, i.e., all Project names

    def get_all_cpp_files(self, PathStr: str): # --PathStr = /home/wenxinyao/repository
        for projectStr in self.projects:
            files_list = []  # Store all .cpp files in the current project
            for root, dirs, files in os.walk(PathStr + '/' + projectStr):
                if len(files) > 0:
                    for file_ in files:
                        if '.cpp' in file_ or '.cc' in file_ and 'Test' not in file_ and 'test' not in file_:
                            files_list.append(root + '/' + file_)
            self.pro_file_list.append(files_list)  # Add the .cpp file path list to the main list


class getASTFunctions:
    '''This class is used to convert C++ files to AST using clang.index and extract function nodes'''
    def __init__(self, cppFile: str, funcASTs: List) -> None:
        self.cppFile = cppFile  # Path of the C++ file to process
        self.funcASTs = funcASTs  # Store function AST nodes in the .cpp file
        self.funcSrcs = [] # Store function source code in the .cpp file
        self.funcType = [CursorKind.FUNCTION_DECL, CursorKind.CLASS_TEMPLATE,
                         CursorKind.FUNCTION_TEMPLATE, CursorKind.CXX_METHOD, CursorKind.CLASS_DECL]

    def trans_file_to_AST(self):  # Use clang to convert .cpp file to AST
        try:
            index = clang.cindex.Index.create()
            translation_unit = index.parse(self.cppFile)
        except clang.cindex.TranslationUnitLoadError:
            translation_unit = None
        print("tu:", translation_unit)
        return translation_unit

    def get_func_ASTs(self, translation_unit):  # Extract function declaration AST nodes
        funcCurs = []  # Store function node pointers in the .cpp file
        processed_ranges = []  # Used to record the ranges of processed functions (filter nested functions)
        if translation_unit is not None:
            node_list = list(translation_unit.cursor.get_children())
            i = 0  # Use index to traverse node_list
            while i < len(node_list):
                node = node_list[i]
                if node.location.file and node.location.file.name == self.cppFile:
                    if node.kind in self.funcType:
                        current_range = (node.extent.start.line, node.extent.end.line)
                        is_nested = any( # Check if the current function range overlaps with processed ranges
                            start_line <= current_range[0] and current_range[1] <= end_line
                            for start_line, end_line in processed_ranges
                        )
                        if not is_nested: # If not overlapping, record the function
                            funcCurs.append(node)
                            processed_ranges.append(current_range)  # Record the line range of the current function
                    elif node.kind == CursorKind.NAMESPACE: # Also add functions under the namespace to funcCurs (next level)
                        node_list.extend(node.get_children())
                i += 1
        else:
            funcCurs = []
        
        for cur in funcCurs:
            print(f"function:{cur}")
            node_info_list = []
            count = self.traverse_ast(cur, node_info_list) # Traverse all child nodes of the function node and store related information
            # print(f"{count} == {len(node_info_list)}")
            self.funcASTs.append(node_info_list)
            funcs_snippet = self.get_code_snippet(cur.extent) # Source code corresponding to the function
            self.funcSrcs.append(funcs_snippet)


    # Function to traverse AST nodes and output depth information
    def traverse_ast(self, node, node_info_list, depth=0, count=0):
        if node.location.file and node.location.file.name == self.cppFile:
            try:
                count += 1
                code_snippet = self.get_code_snippet(node.extent)
                node_info = { # Extract node information and store in a dictionary
                    "depth": depth,
                    "extent": f"{node.extent.start.line}-{node.extent.start.column}-{node.extent.end.line}-{node.extent.end.column}",
                    "kind": str(node.kind),
                    "type_kind": str(node.type.kind) if node.type.kind else None,
                    "type_spelling": node.type.spelling if node.type.spelling else None,
                    "spelling": node.spelling if node.spelling else None,
                    "displayname": node.displayname if node.displayname and node.displayname != node.spelling else None,
                    "code_snippet": code_snippet
                }
                node_info_list.append(node_info)
            except (ValueError, AttributeError) as e: # Catch exceptions, skip the current node, continue processing
                print(f"Warning: Skipping node due to error: {e}")

        for child in node.get_children():
            count = self.traverse_ast(child, node_info_list, depth + 1, count)
        
        return count # count == len(node_info_list)


    def get_code_snippet(self, extent):
        """Extract code content from the source file according to the start and end line/column of the node, and remove comments"""
        try:
            with open(self.cppFile, "r", encoding="utf-8") as file:
                lines = file.readlines()
                # Get start and end line content
                start_line_content = lines[extent.start.line - 1]  # Start line
                end_line_content = lines[extent.end.line - 1]  # End line
                # If start and end lines are the same, directly extract the content in the column range
                if extent.start.line == extent.end.line:
                    snippet = start_line_content[extent.start.column - 1:extent.end.column - 1]
                    return self.remove_extra_newlines(self.remove_single_line_comments(snippet)).strip()
                # If start and end lines are different, process and concatenate code fragments line by line
                snippet = self.remove_single_line_comments(start_line_content[extent.start.column - 1:]) + "\n"  # Extract start line
                for line_num in range(extent.start.line, extent.end.line - 1):
                    line_content = lines[line_num]
                    if re.match(r"^\s*//", line_content):  # If the line is only a comment, skip
                        continue
                    snippet += self.remove_single_line_comments(line_content) + "\n"  # Add middle lines
                snippet += self.remove_single_line_comments(end_line_content[:extent.end.column - 1])  # Extract end line
                return self.remove_extra_newlines(self.remove_multi_line_comments(snippet)).strip()
        except Exception as e:
            print(f"Cannot read file {self.cppFile}: {e}")
            return None


    def remove_single_line_comments(self, code):
        """Remove single-line comments from // to the end of the line"""
        pattern = r"//.*?$"
        cleaned_code = re.sub(pattern, "", code, flags=re.MULTILINE)
        return cleaned_code


    def remove_multi_line_comments(self, code):
        """Remove multi-line comments /* ... */"""
        pattern = r"/\*.*?\*/"
        cleaned_code = re.sub(pattern, "", code, flags=re.DOTALL)
        return cleaned_code


    def remove_extra_newlines(self, code):
        """Remove multiple consecutive newlines, keep only one"""
        pattern = r"\n\s*\n"
        cleaned_code = re.sub(pattern, "\n", code)
        return cleaned_code


    def print_func_ASTs(self): # Print the extracted function AST nodes
        for node in self.funcASTs:
            print(node)


def filter_no_func_file(GCF: GetCppFile, pro_func_ast: List[clang.cindex.Cursor], pro_func_ast_src):
    '''
    Filter files without valid functions, return file paths and AST nodes containing valid functions
    pro_func_ast stores all function AST nodes in all Projects (3D list)
    '''
    pro_files = []  # Store .cpp file paths containing valid functions in all projects
    pro_funcs = []  # Store function AST nodes contained in all projects
    pro_funcs_src = []
    for i in range(len(GCF.projects)):  # Traverse each project
        pro_files_ = []  # Store .cpp file paths containing valid functions
        pro_funcs_ = []  # Store lists containing valid function AST nodes
        pro_funcs_src_ = []
        for j in range(len(pro_func_ast[i])):  # Traverse each .cpp file in the project
            if pro_func_ast[i][j] is not None and len(pro_func_ast[i][j]) != 0:
                pro_files_.append(GCF.pro_file_list[i][j])
                pro_funcs_.append(pro_func_ast[i][j])
                pro_funcs_src_.append(pro_func_ast_src[i][j])
        pro_files.append(pro_files_)
        pro_funcs.append(pro_funcs_)
        pro_funcs_src.append(pro_funcs_src_)
    return pro_files, pro_funcs, pro_funcs_src


def write_data(GCF: GetCppFile, pro_func_ast: List[clang.cindex.Cursor], pro_func_ast_src, dataFilePath: str):
    pro_files, pro_funcs, pro_funcs_src = filter_no_func_file(GCF, pro_func_ast, pro_func_ast_src)
    # DataFrame is expanded into rows by the first dimension, i.e., each row stores a quadruple of a Project
    data = pd.DataFrame({'project': GCF.projects, 'cppFile': pro_files, 'func_ast': pro_funcs, 'func_src': pro_funcs_src})
    print(data.head())
    print("Columns:", data.columns, "Shape:", data.shape)
    pd.to_pickle(data, dataFilePath)


def read_data(dataFilePath: str):
    data = pd.read_pickle(dataFilePath)
    return data


def main():
    clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang-10.so.1')
    projects = []  # Initialize GCF.projects
    data_files = []  # Initialize GCF.pro_file_list
    pro_func_ast = []  # Store all function ASTs in each project (3D list)
    pro_func_ast_src = []

    parser = argparse.ArgumentParser(description='args description')
    parser.add_argument('--PathStr', '-PS', default='../../repository', help='input project path')
    parser.add_argument('--dataFilePath', '-dFP', default='../files/dataset.pkl', help='output data file path')
    args = parser.parse_args()

    GCF = GetCppFile(projects, data_files)
    GCF.get_cpp_projects(args.PathStr)  # Get GCF.projects
    GCF.get_all_cpp_files(args.PathStr)  # Get GCF.pro_file_list

    for i in range(len(GCF.projects)):  # Traverse the i-th Project
        funcs_in_pro = [] # Store all function AST nodes in the current Project (2D list)
        funcs_in_pro_src = []
        for j in range(len(GCF.pro_file_list[i])): # Traverse the j-th .cpp file in the i-th Project
            if (i == 98 and j == 651): # Exclude error file
                funcs_in_pro.append(None)
                funcs_in_pro_src.append(None)
                continue
            pro_file_tmp = GCF.pro_file_list[i][j]
            print(f"[{i}, {j}]: [{GCF.projects[i]}, {pro_file_tmp}]")
            funcASTs = []  # Initialize AST parser for each .cpp file
            Parser = getASTFunctions(pro_file_tmp, funcASTs)
            translation_unit = Parser.trans_file_to_AST()  # Get AST of .cpp file
            Parser.get_func_ASTs(translation_unit) # Get Parser.funcASTs, store all function AST nodes of the current .cpp file
            # Parser.print_func_ASTs()
            funcs_in_pro.append(Parser.funcASTs) # Current .cpp file -> Current Project
            funcs_in_pro_src.append(Parser.funcSrcs)
        pro_func_ast.append(funcs_in_pro) # Current Project -> All Projects
        pro_func_ast_src.append(funcs_in_pro_src)
    
    write_data(GCF, pro_func_ast, pro_func_ast_src, args.dataFilePath)


def extract_dataset(folder_path, out_dir):
    clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang-10.so.1')
    projects = []  # Initialize GCF.projects
    data_files = []  # Initialize GCF.pro_file_list
    pro_func_ast = []  # Store all function ASTs in each project (3D list)
    pro_func_ast_src = []

    GCF = GetCppFile(projects, data_files)
    GCF.get_cpp_projects(folder_path)  # Get GCF.projects
    GCF.get_all_cpp_files(folder_path)  # Get GCF.pro_file_list

    for i in range(len(GCF.projects)):  # Traverse the i-th Project
        funcs_in_pro = [] # Store all function AST nodes in the current Project (2D list)
        funcs_in_pro_src = []
        for j in range(len(GCF.pro_file_list[i])): # Traverse the j-th .cpp file in the i-th Project
            if (i == 98 and j == 651): # Exclude error file
                funcs_in_pro.append(None)
                funcs_in_pro_src.append(None)
                continue
            pro_file_tmp = GCF.pro_file_list[i][j]
            print(f"[{i}, {j}]: [{GCF.projects[i]}, {pro_file_tmp}]")
            funcASTs = []  # Initialize AST parser for each .cpp file
            Parser = getASTFunctions(pro_file_tmp, funcASTs)
            translation_unit = Parser.trans_file_to_AST()  # Get AST of .cpp file
            Parser.get_func_ASTs(translation_unit) # Get Parser.funcASTs, store all function AST nodes of the current .cpp file
            # Parser.print_func_ASTs()
            funcs_in_pro.append(Parser.funcASTs) # Current .cpp file -> Current Project
            funcs_in_pro_src.append(Parser.funcSrcs)
        pro_func_ast.append(funcs_in_pro) # Current Project -> All Projects
        pro_func_ast_src.append(funcs_in_pro_src)
    
    write_data(GCF, pro_func_ast, pro_func_ast_src, out_dir)


# nohup python3 repo2data.py --PathStr ../../repository --dataFilePath ../files/dataset-test.pkl > repo2data.log 2>&1 &
if __name__ == "__main__":
    main()