from swebench.harness.utils import load_swebench_dataset
import json
import os
from tqdm.auto import tqdm
import json
import pandas as pd
import glob
instances = load_swebench_dataset(split="test", name="princeton-nlp/SWE-bench_Verified")
final_patch_dir = "/Users/shrey/Desktop/Composio-dev/composio/python/swe/agentic/langgraph_agent/logs/run_evaluation/langgraph_agent_final2/composio"

def get_trace(dfs, start_id, end_id):    
    trace = []
    for i in range(end_id, start_id - 1, -1):
        df = dfs[i]
        if len(df) == 1:
            trace.append(df["inputs"][0]["messages"] + [df["outputs"][0]])
        else:
            trace.append(df.iloc[-1]["inputs"]["messages"] + [df.iloc[-1]["outputs"]])
    return trace

def find_start_end(dfs, lens):
    start = -1
    for i, df in enumerate(dfs):
        if "You are a software engineer expert at solving bugs" in df.loc[0, "inputs"]["messages"][0]['kwargs']['content']:
            start = i
            break
    lens_copy = lens[start:]
    begin = -1
    end = -1
    for i, value in enumerate(lens_copy):
        if value != 1:
            if begin == -1:
                begin = i
            end = i
        elif begin != -1:
            break
    end = min(end, begin+2)
    end+=start
    return start, end

def get_final_patches(final_patch_dir):
    final_patches = {}
    for patch_path in glob.glob(final_patch_dir + "/**/patch.diff"):
        project_name = patch_path.split("/")[-2]
        final_patches[project_name] = open(patch_path, "r").read()
    return final_patches

final_patches = get_final_patches(final_patch_dir)

def get_traces_from_folder(folder_path):
    run_dict = {}

    for x in tqdm(instances):
        project_name = x["instance_id"]
        if os.path.exists(f"./logs/{folder_path}/{project_name}/runs.json"):
            with open(f"./logs/{folder_path}/{project_name}/runs.json", "r") as f:
                runs_dict = json.load(f)
                df = pd.DataFrame(runs_dict)
                run_dict[project_name] = df

    for k,df in run_dict.items():
        dfs = []
        for _, _df in df.groupby(by="trace_id"):
            _df = _df.sort_values(by="start_time", ascending=True) 
            dfs.append(_df.reset_index(drop=True))
        dfs.sort(key=lambda x: x.iloc[0]['start_time'], reverse=True)
        run_dict[k] = dfs

    for k, dfs in run_dict.items():
        filtered_dfs = []
        for _df in dfs:
            if _df.iloc[0]["name"] == "BedrockChat": filtered_dfs.append(_df)
        if len(filtered_dfs) == 0:
            # print(k)
            pass
        else:
            run_dict[k] = filtered_dfs
    
    trace = {}
    probs = []
    for k, dfs in run_dict.items():
        lens = [len(_df) for _df in dfs]
        if k == "django__django-16901":
            start, end = 7, 13
        elif k == "sympy__sympy-11618":
            start, end = 7, 13
        else:
            start, end = find_start_end(dfs, lens)
        if start == -1:
            probs.append(k)
            continue
        if not (final_patches[k] in dfs[start].loc[0, "inputs"]["messages"][1]["kwargs"]["content"]):
            probs.append(k)
            continue
        tr = get_trace(dfs, start, end)
        trace[k] = tr

    return run_dict, trace, probs

final_trace = {}
for traj_path in glob.glob("./logs/trajs/*.json"):
    key = traj_path.split("/")[-1][:-10]
    traj = json.load(open(traj_path, 'r'))
    final_trace[key] = traj

for folder in ["runs", "runs2", "runs3"]:
    run_dict, trace, probs = get_traces_from_folder(folder)
    final_trace.update(trace)

problematic = []
for key, patch in final_patches.items():
    if patch not in final_trace[key][-1][1]['kwargs']['content']:
        problematic.append(key)

print("Problematic instances:")
for key in problematic:
    print(key)

a = input("Do you want to go ahead and save the traces? (y/n): ")

if a == "y":
    for key, traj in final_trace.items():
        with open(f"./logs/trajs/{key}_traj.json", "w") as f:
            json.dump(traj, f, indent=2)


