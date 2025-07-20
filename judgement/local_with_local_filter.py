# First determine whether all frequent code fragments are idioms, then take the top n for synthesis
import os
import openai
from openai import OpenAI
import pandas as pd
from difflib import SequenceMatcher
from .prompt_set import idiom_judge_prompt, idiom_synthesis_prompt
import traceback
import argparse
import time
import sys
from .utils import Utils
import pickle
from dataclasses import dataclass
import copy

@dataclass
class LocLabelInfo:
    pro_name: str      # Project name
    loc_label: str     # Code location label
    is_train: bool     # Whether it belongs to the training set
    total_num: int     # Total number of nodes (len(func))
    idiom_num: int     # Number of nodes that are idioms

def get_pattern_result(key, model, base_url, cluster_data, completion_result_file, k, test_mode=False, loc_info_list=None):
    """This function reads clustering data (cluster_data.pkl), analyzes the center point of each cluster, and generates information about code patterns."""
    # cluster_data = pd.read_pickle(cluster_data_file)
    cluster_labels = cluster_data['loc_label'] # Location labels
    cluster_center_points_all = cluster_data['center_point'] # Frequent sub-idiom code fragments
    cluster_sizes = cluster_data['cluster_size']
    cluster_center_point_infos = cluster_data['center_point_info'] # Frequent sub-idiom detailed info (pro_name, file_name, extent_root, node_info(dict))
    sub_infos = cluster_data['infos'] # All sub-idiom info
    print(f"Clustering data loaded, length: {len(cluster_labels)}-{len(cluster_center_points_all)}-{len(cluster_sizes)}-{len(cluster_center_point_infos)}-{len(sub_infos)}")

    # If in test mode, only take the first 10 clustering results
    if test_mode:
        cluster_labels = cluster_labels[:10]
        cluster_center_points_all = cluster_center_points_all[:10]
        cluster_sizes = cluster_sizes[:10]
        cluster_center_point_infos = cluster_center_point_infos[:10]
        sub_infos = sub_infos[:10]
    print(f"Test filtering done, length: {len(cluster_labels)}-{len(cluster_center_points_all)}-{len(cluster_sizes)}-{len(cluster_center_point_infos)}-{len(sub_infos)}")
    sys.stdout.flush()

    # Define the final idiom list (real idiom, sub-idiom 1, sub-idiom 2, LLM judgement result, iteration label, included sub-idiom id list, in_train, in_test)
    completion_result = pd.DataFrame(columns=['pattern_completion', 'sgl_idiom', 'iter_idiom', 'final_completion', 'label', 'ids', 'in_train', 'in_test'])

    def update_in_train_and_in_test(sgl_idiom, iter_idiom):
        """
        Update the new_idiom's in_train and in_test labels according to sgl_idiom and iter_idiom.
        If either is 1, the corresponding label of new_idiom is also 1.
        """
        # Initialize labels
        sgl_in_train, sgl_in_test = 0, 0
        iter_in_train, iter_in_test = 0, 0

        # Find sgl_idiom's labels
        for _, row in completion_result.iterrows():
            if row['pattern_completion'] == sgl_idiom:
                sgl_in_train = row['in_train']
                sgl_in_test = row['in_test']
                break

        # Find iter_idiom's labels
        for _, row in completion_result.iterrows():
            if row['pattern_completion'] == iter_idiom:
                iter_in_train = row['in_train']
                iter_in_test = row['in_test']
                break

        # Compute new_idiom's labels
        new_in_train = 1 if sgl_in_train == 1 or iter_in_train == 1 else 0
        new_in_test = 1 if sgl_in_test == 1 or iter_in_test == 1 else 0

        return new_in_train, new_in_test

    def single_pattern_result(cur_labels, cur_idioms, cur_sizes, cur_idiom_infos, cur_sub_infos):
        """
        For each cluster center, generate analysis results of code patterns, and call GPT to evaluate whether it is a code pattern and whether it fits real application scenarios.
        loc_label: cur_idioms frequent sub-idiom code fragments, cur_idiom_infos frequent sub-idiom detailed info
        """
        try:
            i = 0
            while i < len(cur_idioms): # Traverse the i-th frequent sub-idiom
                # if i == 3: # Only process the first 3 frequent sub-idioms
                #     break
                cur_idiom = cur_idioms[i]
                print(f"The {i}th frequent sub-idiom: {cur_idiom}")
                if Utils.code_filter(cur_idiom):
                    print("Filter :(", cur_idiom)
                    cur_labels.pop(i)
                    cur_idioms.pop(i)  # Remove current element from list
                    cur_sizes.pop(i)
                    cur_idiom_infos.pop(i)
                    cur_sub_infos.pop(i)
                    continue
                final_completion = final_question(key, model, base_url, cur_idiom)
                # If there is no 'no' in the final evaluation by GPT, the frequent sub-idiom can be regarded as an idiom and added to the idiom list
                final_completion_str = final_completion['choices'][0]['message']['content']
                if 'no' not in final_completion_str[:5].lower() and not Utils.contains_string(completion_result['pattern_completion'].tolist(), cur_idiom):
                    print("cur_idiom_infos[i]", cur_idiom_infos[i])
                    in_train, in_test = process_is_train_and_test(cur_idiom_infos[i], cur_sub_infos[i], loc_info_list)
                    completion_result.loc[len(completion_result.index)] = [cur_idiom, None, None, final_completion_str, 1, [i], in_train, in_test]
                    print(":)", final_completion_str)
                    i += 1  # Only increment index when successfully processed
                else:
                    print(":(", final_completion_str)
                    cur_labels.pop(i)
                    cur_idioms.pop(i)  # Remove current element from list
                    cur_sizes.pop(i)
                    cur_idiom_infos.pop(i)
                    cur_sub_infos.pop(i)
                sys.stdout.flush()
            return cur_labels, cur_idioms, cur_sizes, cur_idiom_infos, cur_sub_infos
        except:
            traceback.print_exc()

    def iter_pattern_result(sgl_idioms, sgl_idiom_infos, sgl_idiom_ids, combine_result_j, combine_result_info_j, combine_result_id_j, j):
        """
        Try to merge similar cluster centers to generate new code fragments, use SequenceMatcher to find similar code fragments and synthesize,
        and use GPT to verify whether these synthesized code fragments conform to code patterns.
        combine_result_j = combine_result[j] frequent sub-idiom code fragments (increasing with synthesis)
        combine_result_info_j = combine_result_info[j] frequent sub-idiom detailed info (increasing with synthesis)
        """
        nxt_iter_result_j = []
        nxt_combine_result_info_j = []
        nxt_combine_result_id_j = []
        try:
            for i in range(len(combine_result_j)): # Traverse each newly synthesized frequent sub-idiom
                assert len(combine_result_j) == len(combine_result_info_j), "Two lists have different lengths"
                for x in range(len(sgl_idioms)): # Traverse each initial single frequent sub-idiom
                    if j == 0 and i >= x: # The 0th iteration is self-iteration, ensure synthesis index i<x
                        continue
                    # SequenceMatcher is a general sequence comparator, compares the similarity of two frequent sub-idioms
                    # m = SequenceMatcher(None, cur_idioms[x], combine_result_j[i])
                    # Use greedy algorithm to find the longest matching subsequence between two sequences, try to synthesize if length > 5 and not exactly the same
                    # match = m.find_longest_match(0, len(cur_idioms[x]), 0, len(combine_result_j[i]))
                    sgl_idiom = sgl_idioms[x]
                    iter_idiom = combine_result_j[i]
                    if Utils.compare_contain_strings(sgl_idiom, iter_idiom): # The two code fragments are not exactly the same
                        new_ids = Utils.merge_lists(sgl_idiom_ids[x], combine_result_id_j[i])
                        if new_ids == [-1]: # There are duplicate elements in the two id lists
                            print(f"Duplicate elements in two id lists: {repr(sgl_idiom)} <--> {repr(iter_idiom)}")
                            continue
                        new_set = set(new_ids) # Convert new_ids to set to ignore order
                        if any(set(existing_list) == new_set for existing_list in nxt_combine_result_id_j): # Duplicate id list
                            print(f"This idiom has been synthesized: {new_set}")
                            continue
                        # Try to synthesize, call GPT to determine whether the synthesized frequent sub-idiom is valid
                        new_idiom = get_synthesized_pattern(key, model, base_url, sgl_idiom, iter_idiom)
                        final_completion = final_question(key, model, base_url, new_idiom)
                        # If there is no 'no' in the final evaluation by GPT, the frequent sub-idiom can be regarded as an idiom and added to the idiom list
                        final_completion_str = final_completion['choices'][0]['message']['content']
                        print(f"Try to synthesize {i}-{x}: {repr(sgl_idiom)} <--> {repr(iter_idiom)}")
                        if 'no' not in new_idiom[:5].lower() and 'no' not in final_completion_str[:5].lower():
                            if not Utils.contains_string(completion_result['pattern_completion'].tolist(), new_idiom): # Avoid duplicate addition
                                # Update the synthesized frequent sub-idiom info, as input for the next iteration
                                nxt_iter_result_j.append(new_idiom) # Add synthesized idiom code fragment
                                new_ast_num = combine_result_info_j[i][3].get("ast_num", 0) + sgl_idiom_infos[x][3].get("ast_num", 0) # Calculate node count after synthesis
                                print(f'{combine_result_info_j[i][3].get("ast_num", 0)} + {sgl_idiom_infos[x][3].get("ast_num", 0)} = {new_ast_num}')
                                new_info = copy.deepcopy(combine_result_info_j[i]) # Cannot use shallow copy
                                new_info[3]["ast_num"] = new_ast_num # Update node count after synthesis
                                nxt_combine_result_info_j.append(new_info) # combine_result_info_j[i] as info after synthesis (for later node count calculation)
                                nxt_combine_result_id_j.append(new_ids) # Update id list after synthesis
                                in_train, in_test = update_in_train_and_in_test(sgl_idiom, iter_idiom) # Update in_train and in_test after synthesis
                                # Add to the project's final idiom list
                                completion_result.loc[len(completion_result.index)] = [new_idiom, sgl_idiom, iter_idiom, final_completion_str, j + 2, new_ids, in_train, in_test]
                                print(f'Valid synthesized idiom: {new_idiom} <-- {final_completion_str}')
                                sys.stdout.flush()
        except:
            traceback.print_exc()
        return nxt_iter_result_j, nxt_combine_result_info_j, nxt_combine_result_id_j

    
    average_size_list = []
    print(f"Before judgement: {len(cluster_labels)}-{len(cluster_center_points_all)}-{len(cluster_sizes)}-{len(cluster_center_point_infos)}-{len(sub_infos)}")
    
    # Single pattern recognition, directly recognize frequent sub-idioms obtained after clustering as real idioms (later is recognition after k iterations)
    new_labels, new_idioms, new_sizes, new_idiom_infos, new_sub_infos = single_pattern_result(
        cluster_labels.tolist(),
        cluster_center_points_all.tolist(),
        cluster_sizes.tolist(),
        cluster_center_point_infos.tolist(),
        sub_infos.tolist()
    )

    # Replace original variables with processed results
    cluster_labels = pd.Series(new_labels)
    cluster_center_points_all = pd.Series(new_idioms)
    cluster_sizes = pd.Series(new_sizes)
    cluster_center_point_infos = pd.Series(new_idiom_infos)
    sub_infos = pd.Series(new_sub_infos)

    print(f"After judgement: {len(cluster_labels)}-{len(cluster_center_points_all)}-{len(cluster_sizes)}-{len(cluster_center_point_infos)}-{len(sub_infos)}")
    # Sort by cluster_sizes descending, take top 300
    sorted_indices = cluster_sizes.argsort()[::-1][:300]
    cluster_labels = cluster_labels.iloc[sorted_indices].reset_index(drop=True)
    cluster_center_points_all = cluster_center_points_all.iloc[sorted_indices].reset_index(drop=True)
    cluster_sizes = cluster_sizes.iloc[sorted_indices].reset_index(drop=True)
    cluster_center_point_infos = cluster_center_point_infos.iloc[sorted_indices].reset_index(drop=True)
    sub_infos = sub_infos.iloc[sorted_indices].reset_index(drop=True)
    print(f"After sorting: {len(cluster_labels)}-{len(cluster_center_points_all)}-{len(cluster_sizes)}-{len(cluster_center_point_infos)}-{len(sub_infos)}")

    # Divide the dataset into several classes according to labels
    unique_labels = cluster_labels.unique()
    for label in unique_labels:
        # Extract all idiom data for the current location label
        label_list = label.split('-')
        print(f"Processing current location label: {label_list[1]} {label_list[-4:]}")
        label_indices = cluster_labels[cluster_labels == label].index
        sgl_idioms = cluster_center_points_all[label_indices].tolist()
        sgl_idiom_infos = cluster_center_point_infos[label_indices].tolist()
        sgl_sub_infos = sub_infos[label_indices].tolist()
        print(f"Number of idioms for current location label: {len(label_indices)}-{len(sgl_idioms)}-{len(sgl_idiom_infos)}-{len(sgl_sub_infos)}")

        if len(sgl_idioms) == 0: # If there are no idioms, skip this label
            continue
        sgl_idiom_ids = [[i] for i in range(len(sgl_idioms))] # Indices of frequent sub-idioms
        print(f"Single pattern recognition done, current idiom count: {len(sgl_idioms)}-{len(sgl_idiom_infos)}-{len(sgl_sub_infos)}")
        sys.stdout.flush()

        # Traverse first-level list sgl_idiom_infos
        for info in sgl_idiom_infos:
            ast_num = info[3].get("ast_num", 0)  # Get ast_num

            # Find the corresponding element in loc_info_list and update idiom_num
            for loc_info in loc_info_list:
                if loc_info.loc_label == label:
                    loc_info.idiom_num += ast_num
                    break

        # Traverse second-level list sgl_sub_infos
        for sub_list in sgl_sub_infos:
            for sub_info in sub_list:
                ast_num = sub_info[3].get("ast_num", 0)  # Get ast_num
                loc_label = f"{sub_info[0]}-{sub_info[1]}-{sub_info[2]}"  # Concatenate pro_name, file_name, extent_root as loc_label

                # Find the corresponding element in loc_info_list and update idiom_num
                for loc_info in loc_info_list:
                    if loc_info.loc_label == loc_label:
                        loc_info.idiom_num += ast_num
                        break
        
        for loc_info in loc_info_list:
            if loc_info.idiom_num > 0:
                rate = (float)(loc_info.idiom_num) / loc_info.total_num if loc_info.idiom_num < loc_info.total_num else 1.0
                print(f"{loc_info.total_num} -> {loc_info.idiom_num} : {rate} {loc_info.loc_label}")

        # Initialize synthesis results, copy single idioms
        combine_result = [sgl_idioms] # Frequent sub-idiom code fragments (increasing with synthesis)
        combine_result_info = [sgl_idiom_infos] # Frequent sub-idiom detailed info (increasing with synthesis)
        combine_result_id = [sgl_idiom_ids] # Indices of frequent sub-idioms (increasing with synthesis)

        # When k=3, iterate 3 times (max synthesis count is 3), each time try to synthesize idioms in completion_result (SequenceMatcher match value)
        for j in range(k):
            print(f"Iteration {j} start, current input idiom count: {len(combine_result[j])}")
            # Each time, the input is the synthesis result of the previous time, and in the next time, continue to synthesize with the initial single frequent sub-idioms
            nxt_iter_result_j, nxt_combine_result_info_j, nxt_combine_result_id_j  = iter_pattern_result(sgl_idioms, sgl_idiom_infos, sgl_idiom_ids, combine_result[j], combine_result_info[j], combine_result_id[j], j)
            combine_result.append(nxt_iter_result_j) # Frequent sub-idiom code fragments (increasing with synthesis)
            combine_result_info.append(nxt_combine_result_info_j) # Frequent sub-idiom detailed info (increasing with synthesis)
            combine_result_id.append(nxt_combine_result_id_j) # Indices of frequent sub-idioms (increasing with synthesis)
            print(f"Iteration {j} done, current idiom count: {len(nxt_iter_result_j)}, output synthesized idioms:\n{nxt_iter_result_j}")
            sys.stdout.flush()

        # Calculate the average size of all idioms according to combine_result_info
        flattened_info = [item for sublist in combine_result_info for item in sublist]  # Flatten 2D list to 1D
        avg_size = sum(info[3].get("ast_num", 0) for info in flattened_info) / len(flattened_info)  # Calculate average
        average_size_list.append(avg_size)
        print(f"Current average size: {avg_size}")


    pro_name = cluster_center_point_infos[0][0]  # Get project name
    pro_avg_size = round(sum(average_size_list) / len(average_size_list), 2)
    print(f"{pro_name}'s avg_size : {pro_avg_size}")
    pro_ic = round(calculate_coverage(loc_info_list), 2)
    print(f"{pro_name}'s ic : {pro_ic}")
    both_true_count = len(completion_result[(completion_result['in_train'] == 1) & (completion_result['in_test'] == 1)])
    in_train_count = len(completion_result[completion_result['in_train'] == 1])
    pro_isp = round(both_true_count / in_train_count, 2)
    print(f"{pro_name}'s isp : {pro_isp}")
    pro_f1 = round(2 * (pro_ic * pro_isp) / (pro_ic + pro_isp), 2) if pro_ic + pro_isp > 0 else 0.00
    print(f"{pro_name}'s f1 : {pro_f1}")

    # Save the final idiom list
    pd.to_pickle(completion_result, completion_result_file)

    return pro_avg_size, pro_ic, pro_isp, pro_f1


def process_is_train_and_test(cur_idiom_info, cur_sub_info, loc_info_list):
    """
    Concatenate loc_label from cur_idiom_info and cur_sub_info, and find the corresponding is_train attribute in loc_info_list.
    Return the values of in_train and in_test.
    """
    loc_label_list = []

    # Concatenate loc_label from cur_idiom_info
    loc_label = f"{cur_idiom_info[0]}-{cur_idiom_info[1]}-{cur_idiom_info[2]}"
    loc_label_list.append(loc_label)

    # Traverse cur_sub_info, concatenate loc_label
    for sub_info in cur_sub_info:
        sub_loc_label = f"{sub_info[0]}-{sub_info[1]}-{sub_info[2]}"
        loc_label_list.append(sub_loc_label)

    # Initialize in_train and in_test
    in_train = 0
    in_test = 0

    # Traverse loc_info_list, find is_train attribute according to loc_label_list
    for loc_label in loc_label_list:
        for loc_info in loc_info_list:
            if loc_info.loc_label == loc_label:
                if loc_info.is_train:
                    in_train = 1
                else:
                    in_test = 1

    return in_train, in_test


# Calculate coverage
def calculate_coverage(loc_info_list):
    total_coverage = 0
    valid_count = 0

    for loc_info in loc_info_list:
        if loc_info.idiom_num > 0:  # Calculate coverage when idiom_num is not 0
            coverage = (float)(loc_info.idiom_num) / loc_info.total_num if loc_info.idiom_num < loc_info.total_num else 1.0
            total_coverage += coverage
            valid_count += 1

    # Calculate average coverage
    pro_idiom_coverage = total_coverage / valid_count
    return pro_idiom_coverage

def final_question(key, model, base_url, completion_answer):
    """Determine whether the code can be regarded as an idiom"""
    messages = idiom_judge_prompt.copy()
    user_prompt = f"Determine whether the Code Fragment [{completion_answer}] is a legitimate code idiom."
    messages.append({"role": "user", "content": user_prompt})
    return call_openai_api(key, model, base_url, messages)


def get_synthesized_pattern(key, model, base_url, idiom1, idiom2):
    """Synthesize two code fragments, parameters are two frequent sub-idiom code fragments"""
    messages = idiom_synthesis_prompt.copy()
    user_prompt = f"Integrate Code Fragment 1 [{idiom1}] and Code Fragment 2 [{idiom2}] into a reasonable Code Fragment."
    messages.append({"role": "user", "content": user_prompt})

    completion = call_openai_api(key, model, base_url, messages)
    if completion:
        new_idiom = completion["choices"][0]["message"]["content"]
        return new_idiom
    return None


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


def syhthesize_by_project(key, model, base_url):
    cluster_results_file = "../files/gcb_cluster_results.pkl"
    cluster_results = pd.read_pickle(cluster_results_file)
    loc_info_file_path = '../files/loc_info.pkl'
    loc_info_list = pd.read_pickle(loc_info_file_path)

    for project_data in cluster_results:
        pro_name = project_data.get("pros_name")
        if pro_name != "TrafficMonitor":
            continue
        clusters = project_data["clusters"]
        completion_result_file = '../files/l_l_result/' + pro_name + '.pkl'

        # Filter loc_info_list by pro_name
        pro_loc_info_list = [loc_info for loc_info in loc_info_list if loc_info.pro_name == pro_name]
        pro_avg_size, pro_ic, pro_isp, pro_f1 = get_pattern_result(key, model, base_url, clusters, completion_result_file,
                                                               k=3, test_mode=False, loc_info_list=pro_loc_info_list)


def run_judgement_and_synthesis(file_path, key, model, base_url, test_mode):
    cluster_results = pd.read_pickle(file_path)
    loc_info_file_path = 'files/loc_info.pkl'
    loc_info_list = pd.read_pickle(loc_info_file_path)

    for project_data in cluster_results:
        pro_name = project_data.get("pros_name")
        clusters = project_data["clusters"]
        completion_result_file = 'files/' + pro_name + '.pkl'

        pro_loc_info_list = [loc_info for loc_info in loc_info_list if loc_info.pro_name == pro_name]
        get_pattern_result(key, model, base_url, clusters, completion_result_file, k=3, test_mode=test_mode, loc_info_list=pro_loc_info_list)

    return completion_result_file


# nohup python3 local_with_local_filter.py > local_with_local_filter.log 2>&1 &
if __name__ == '__main__': # 806989
    key = 'sk-Amr2bl7ce65IyzQuK3OkACqPDXeg5xF9OrMDIOOQVOqKWOh8'
    model = 'gpt-4o-mini'
    # model = 'DeepSeek-R1'
    base_url = "https://us.ifopen.ai/v1"

    syhthesize_by_project(key, model, base_url)