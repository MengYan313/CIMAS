import pickle
from difflib import get_close_matches
import traceback
import time
import sys
import openai
from prompt_set import standards, standard_prompts
import pandas as pd

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
        print(f"Idioms loaded successfully, number of idioms: {len(first_column)}")
        return first_column
    except Exception as e:
        print(f"Failed to extract the first column: {e}")
        return None


def judge_idiom_standard(key, model, base_url, idiom, standard, standard_prompt):
    """ Determine whether the current idiom meets a certain programming standard """
    messages = standard_prompt.copy()
    user_prompt = f"Does the code idiom [{idiom}] comply with the C++ programming standard: {standard} "
    messages.append({"role": "user", "content": user_prompt})
    completion = call_openai_api(key, model, base_url, messages)
    try:
        content = completion["choices"][0]["message"]["content"]
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


def judge_idiom_standards(key, model, base_url, idiom_source_code_list, output_path):
    """ Determine whether idioms comply with all programming standards """
    results = []
    for i, idiom in enumerate(idiom_source_code_list, start=1):
        print(f"Judging idiom {i}: {idiom}")
        for j, (standard, standard_prompt) in enumerate(zip(standards, standard_prompts), start=1):
            reason = judge_idiom_standard(key, model, base_url, idiom, standard, standard_prompt)
            if "yes" in reason[:5].lower():
                results.append([standard, idiom, reason, 1])
                print(j, "Positive:", reason)
            elif "no" in reason[:5].lower():
                results.append([standard, idiom, reason, 0])
                print(j, "Negative:", reason)
            else:
                print(j, "Irrelevant:", reason)
        sys.stdout.flush()
    
    df = pd.DataFrame(results, columns=["standard", "idiom", "reason", "is_positive"])
    df.to_pickle(output_path)
    print(f"Saved as {output_path}")


def main():
    key = 'sk-Amr2bl7ce65IyzQuK3OkACqPDXeg5xF9OrMDIOOQVOqKWOh8'
    model = 'gpt-4o-mini'
    base_url = "https://us.ifopen.ai/v1"

    pro_name = 'telephony_core_service'
    output_path = '../files/' + pro_name + '_code_standard_idioms.pkl'

    idiom_path = '../files/' + pro_name + '.pkl'
    data = read_pkl_file(idiom_path)
    idiom_source_code_list = extract_first_column(data)
    judge_idiom_standards(key, model, base_url, idiom_source_code_list, output_path)


# nohup python3 2_standards.py > 2_standards.log 2>&1 &
if __name__ == "__main__": # 374061
    main()