import numpy as np
import pandas as pd
import sys

def process_projects(data):
    pros_name = data['pros_name']
    pros_src = data['pros_src']
    pros_info = data['pros_info']

    funcs_type = []
    funcs_name = []
    funcs_decl = []

    for i, pro_name in enumerate(pros_name):
        func_type = []
        func_name = []
        func_decl = []

        print(f"处理项目{i}：{pro_name}")
        pro_src = pros_src[i]
        pro_info = pros_info[i]
        print(len(pro_src), len(pro_info))
        for ele_src, ele_info in zip(pro_src, pro_info):
            node_info = ele_info[-1]
            kind = node_info.get("kind", "")
            spelling = node_info.get("spelling", "")
            if not spelling:
                continue
            
            if kind in ("CursorKind.FUNCTION_DECL", "CursorKind.FUNCTION_TEMPLATE", "CursorKind.CXX_METHOD") and spelling in ele_src:
                # print(f"函数类型 {kind}, 函数名 {spelling}")
                # print(f"函数定义 {ele_src}")
                func_type.append(kind)
                func_name.append(spelling)
                func_decl.append(ele_src)
            
        print(f"项目 {pro_name} 函数数量: {len(func_type)}-{len(func_name)}-{len(func_decl)}")
        funcs_type.append(func_type)
        funcs_name.append(func_name)
        funcs_decl.append(func_decl)
        sys.stdout.flush()
    
    print(f"项目数量：{len(pros_name)}-{len(funcs_type)}-{len(funcs_name)}-{len(funcs_decl)}")
    func_results = pd.DataFrame({'pros_name': pros_name, 'funcs_type': funcs_type, 'funcs_name': funcs_name, 'funcs_decl': funcs_decl})
    return func_results


def main():
    input_file = "../files/embedding.pkl"
    output_file = "../files/func_decl.pkl"

    data = pd.read_pickle(input_file)
    func_results = process_projects(data)
    print(func_results.head())
    print("列名：", func_results.columns, "数据维度：", func_results.shape)
    pd.to_pickle(func_results, output_file)


# nohup python3 get_funcdecl.py > get_funcdecl.log 2>&1 &
if __name__ == "__main__": # 2184354
    main()