import os, json
from glob import glob
import numpy as np, pandas as pd


RESULT_DIR = "./data/adv-glue-plus-plus"
BASE_MODELS = ["alpaca", "vicuna", "stable-vicuna"]


def parse_examples(model):
    # benign_files = glob(os.path.join(RESULT_DIR, "**", "*.json"), recursive=True)
    # target_models = [os.path.relpath(os.path.dirname(x), RESULT_DIR) for x in benign_files]

    df = {
        "BaseModel": [], "TargetModel": [], "Transferability": [], "Accuracy": [], "AccuracyNoRefusal": [],
        "Task": [], "RR+NE": [], "TaskDataCount": []
    }

    failures = {model: {}}
    for target_model in [model]:
        model_file = target_model
        if "hf" in target_model:
            model_file = "".join(target_model.split("hf/")[1:])
        for base_model in BASE_MODELS:
            if not os.path.exists(os.path.join(RESULT_DIR, model_file, f"{base_model}-demo.json")):
                print(f"{os.path.join(RESULT_DIR, model_file, f'{base_model}-demo.json')} does not exist.)")
                continue
            with open(os.path.join(RESULT_DIR, model_file, f"{base_model}-demo.json")) as f:
                j = json.load(f)
                for task in j.keys():
                    if task not in failures[target_model]:
                        failures[target_model][task] = []

                    df["BaseModel"].append(base_model)
                    df["TargetModel"].append(target_model.removeprefix(RESULT_DIR))
                    df["Task"].append(task)
                    df["TaskDataCount"].append(len(j[task]["labels"]))

                    df["Accuracy"].append(
                        np.mean(np.array(j[task]["predictions"]) == np.array(j[task]["labels"]))
                    )

                    df["Transferability"].append(
                        np.mean(np.array(j[task]["predictions"]) != np.array(j[task]["labels"]))
                    )
                    refusal_mask = np.array(j[task]["predictions"]) == -1
                    df["RR+NE"].append(np.mean(refusal_mask))
                    df["AccuracyNoRefusal"].append(
                        np.mean(
                            np.array(j[task]["predictions"])[~refusal_mask] == np.array(j[task]["labels"])[
                                ~refusal_mask]
                        )
                    )
                refusals = {}
                for task in j.keys():
                    preds = j[task]["predictions"]
                    responses = j[task]["responses"]
                    queries = j[task]["requests"]
                    refusals[task] = [

                        y["choices"][0]["message"]["content"] for x, y in zip(preds, responses) if x == -1
                    ]

                    failures[target_model][task].extend(
                        [
                            {
                                "Query": q["messages"][-1]["content"],
                                "Output": y["choices"][0]["message"]["content"]
                            } for q, x, y in zip(queries, preds, responses) if x != y
                        ]
                    )

    return failures


def extract_adv_examples(model, sub_perspective):
    failures = parse_examples(model)
    print(failures[model].keys())
    return failures[model][sub_perspective]


if __name__ == "__main__":
    failure_examples = extract_adv_examples("meta-llama/Llama-2-7b-chat-hf", "mnli")
    print(failure_examples)
