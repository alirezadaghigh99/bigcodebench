
techniques = ["active_to_passive","lowercase_to_uppercase" ,"rephrase_prompt", "verb_to_similar_verb", "declarative_to_interrogative", "adversarial_function_name"]


model = ["gemini"]

ground_truth_test_path = f"generation_output/{model[0]}/test_generation/test_code_original_{model[0]}_bigcodebench_output_llm1.jsonl"
import json, os

ground_truth_data = list(map(json.loads, open(ground_truth_test_path)))
gt_data = {}

for data in ground_truth_data:
    gt_data[data["task_id"]] = {"test_result" : data["test_result"]}


for technique in techniques:
    our_result = f"final_evaluation_results2/{model[0]}/original_vs_{technique}_evaluation.jsonl"
    technique_result = list(map(json.loads, open(our_result)))
    for d in technique_result:
        if d["actual_result"] == 0 and gt_data[d["task_id"]]["test_result"] == 1:
            # print(d["task_id"],d["technique_score"], d["original_score"] )
            if d["technique_score"] > d["original_score"]:
                # print(d["technique_score"], d["original_score"], d["task_id"], technique)

                code_generation_path = f"generation_output/{model[0]}/code_{technique}_{model[0]}_bigcodebench_output_llm1.jsonl"
                

                test_generation_path = f"generation_output/{model[0]}/test_{technique}_{model[0]}_bigcodebench_output_llm1.jsonl"
                
                original_code_generation_path = f"generation_output/{model[0]}/code_original_{model[0]}_bigcodebench_output_llm1.jsonl"
                
                originaltest_generation_path = f"generation_output/{model[0]}/test_original_{model[0]}_bigcodebench_output_llm1.jsonl"
                technique_codes = list(map(json.loads, open(original_code_generation_path)))

                for data in technique_codes:
                    if data["task_id"] == d["task_id"]:
                        print(d["task_id"],d["technique_score"], d["original_score"], technique )
                        # print(data["response_code"])

            