import pickle
import pandas as pd
from difflib import get_close_matches
import traceback
import time
import sys
import openai
import re

def read_pkl_file(file_path):
    """Read pkl file"""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

def extract_first_column(data):
    """Extract the first column of the DataFrame as a list"""
    try:
        first_column = data.iloc[:, 0].tolist()
        return first_column
    except Exception as e:
        print(f"Failed to extract the first column: {e}")
        return None

def find_idioms_with_function(data, function_name, topn=10):
    """
    Retrieve all idioms containing the target function name (case-insensitive).
    If there are more than 10, take the top 10 with the highest similarity. If none, return an empty list.
    """
    try:
        function_name_lower = function_name.lower()
        matching_idioms = [idiom for idiom in data if function_name_lower in str(idiom).lower()]
        idiom_strs = [str(idiom) for idiom in matching_idioms]
        if len(matching_idioms) > topn: # Take the topn with the highest similarity
            matching_idioms = get_close_matches(function_name, idiom_strs, n=topn, cutoff=0.1)
        return matching_idioms
    except Exception as e:
        print(f"Failed to find idioms: {e}")
        return None

def print_matching_idioms(i, func_name, func_decl, matching_idioms):
    """Print matched idioms"""
    if matching_idioms:
        print(f"Function {i}: Related idioms for {func_name}:")
        for idiom in matching_idioms:
            print(f"{idiom}\n")
        print(f"Function definition: {func_decl}")
    else:
        print(f"Function {i}: No related idioms found for {func_name}")

def remove_same_idiom(matching_idioms, func_decl):
    """ Remove elements in matching_idioms that are the same as func_decl (compare after removing all whitespace) """
    def normalize(s):
        return ''.join(str(s).split())
    func_decl_norm = normalize(func_decl)
    return [idiom for idiom in matching_idioms if normalize(idiom) != func_decl_norm] # Only keep different elements

def generate_test_case_for_function(func_name, func_decl, matching_idioms, key, model, base_url, topn=10):
    """
    According to the function name and definition, retrieve related idioms, concatenate the prompt and call the LLM to generate test cases
    """
    # 1. Retrieve related idioms
    idioms_text = "\n\n".join(str(idiom) for idiom in matching_idioms) if matching_idioms else "None"
    # 2. Concatenate prompt
    prompt = (
        f"Please generate unit test code for function {func_name} according to the function definition. You can generate multiple unit tests (no more than 10), and each test code should contain at least one of the given code idioms.\n\n"
        f"[Function Definition]\n{func_decl}\n\n"
        f"[Related Code Idioms]\n{idioms_text}\n\n"
        f"Below, generate unit test code combined with code idioms. To facilitate subsequent string splitting, wrap each unit test function definition with <<< >>>, "
        "for example <<< TEST(XMLUtilTest, ToStrPositiveInteger) { char buf[BUF_SIZE]; int value = 12345; XMLUtil::ToStr(value, buf, BUF_SIZE); ASSERT_STREQ(buf, '12345'); } \n\n"
    )
    messages = [
        {"role": "user", "content": prompt}
    ]
    # 3. Call LLM API
    response = call_openai_api(key, model, base_url, messages)
    try:
        content = response["choices"][0]["message"]["content"]
        print("Generated unit test code:\n", content)
        return content
    except Exception as e:
        print(f"Exception: {e}")
        return "API_ERROR"

def call_openai_api(key, model, base_url, msg):
    """General OpenAI API call function"""
    client = openai.OpenAI(base_url=base_url, api_key=key)
    try:
        response = client.chat.completions.create(model=model, messages=msg)
        # print(response)
        time.sleep(1) 
        return response.to_dict()
    except Exception as e:
        print("API call failed:", e)
        traceback.print_exc()
        return None

def extract_tests_from_string(s):
    """
    Extract all substrings wrapped by <<< >>> and return as a list
    """
    # return re.findall(r'<<<(.*?)>>>', s, re.DOTALL)
    return [x for x in re.findall(r'<<<(.*?)>>>', s, re.DOTALL) if x.strip()]

def process(func_name_list, func_decl_list, idiom_source_code_list, key, model, base_url, output_path):
    print(f"Number of functions {len(func_name_list)}-{len(func_decl_list)}")
    results = []
    
    for i, (func_name, func_decl) in enumerate(zip(func_name_list, func_decl_list)):
        matching_idioms = find_idioms_with_function(idiom_source_code_list, func_name)
        matching_idioms = remove_same_idiom(matching_idioms, func_decl)
        print_matching_idioms(i, func_name, func_decl, matching_idioms)
        if not matching_idioms: # No matching idioms for this function, skip
            continue
        
        content = generate_test_case_for_function(func_name, func_decl, matching_idioms, key, model, base_url)
        unit_test = extract_tests_from_string(content)
        print(f"Number of extracted unit tests: {len(unit_test)}")
        if not unit_test: # Unit test is empty, skip
            continue

        unit_test = unit_test[:10] + [''] * (10 - len(unit_test)) # Pad unit tests to 10
        row = [func_name, func_decl, matching_idioms] + unit_test
        results.append(row)
        sys.stdout.flush()

    columns = ['func_name', 'func_decl', 'matching_idioms'] + [f'unit_test_{i+1}' for i in range(10)]
    df = pd.DataFrame(results, columns=columns)
    df.to_pickle(output_path)
    print("Saved to ", output_path)

def main():
    key = 'sk-Amr2bl7ce65IyzQuK3OkACqPDXeg5xF9OrMDIOOQVOqKWOh8'
    model = 'gpt-4o-mini'
    base_url = "https://us.ifopen.ai/v1"

    func_path = '../files/func_decl.pkl'
    func_data = read_pkl_file(func_path)
    pros_name = func_data['pros_name']
    pro_name = 'TrafficMonitor'
    idx = pros_name[pros_name == pro_name].index[0]
    func_name_list = func_data['funcs_name'][idx]
    func_decl_list = func_data['funcs_decl'][idx]
    output_path = '../files/' + pro_name + '_unitesting.pkl'

    idiom_path = '../files/l_l_result/' + pro_name + '.pkl'
    idiom_data = read_pkl_file(idiom_path)
    idiom_source_code_list = extract_first_column(idiom_data)

    process(func_name_list, func_decl_list, idiom_source_code_list, key, model, base_url, output_path)
    

# nohup python3 1_unitesting.py > 1_unitesting.log 2>&1 &
if __name__ == "__main__": # 2610972
    main()