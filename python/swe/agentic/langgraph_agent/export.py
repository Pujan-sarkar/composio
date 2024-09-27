import os
os.environ['LANGCHAIN_API_KEY']="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"
import json
from langsmith import Client
from uuid import UUID
import pandas as pd
client = Client()
from datetime import datetime, timedelta
from easydict import EasyDict
from swebench.harness.utils import load_swebench_dataset
from tqdm.auto import tqdm

save_dir = "./logs/runs3/"
os.makedirs(save_dir, exist_ok=True)


def get_runs(project_name):
    runs = list(
        client.list_runs(
            project_name=project_name,
            run_type="llm",
        )
    )
    if len(runs) == 0:
        return pd.DataFrame(), []

    runs_df = pd.DataFrame(
        [
            {
                "trace_id": run.trace_id,
                "name": run.name,
                "model": run.extra["invocation_params"]["model_id"] if run.extra and "invocation_params" in run.extra and "model_id" in run.extra["invocation_params"] else None,  
                **run.inputs,
                **(run.outputs or {}),
                "start_time": run.__dict__["start_time"],
                "error": run.error,
                "latency": (run.end_time - run.start_time).total_seconds() if run.end_time else None,  
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "total_tokens": run.total_tokens,
                "metadata": run.metadata,
            }
            for run in runs
        ],
        index=[run.id for run in runs],
    )

    dfs = []
    for _, _df in runs_df.groupby("trace_id"):
        _df_sorted = _df.sort_values(by="start_time")
        dfs.append(_df_sorted)
    
    runs_df = pd.concat(dfs)

    runs_dict = []
    for run in runs:
        run_data = run.__dict__.copy()
        for key, value in run_data.items():
            if isinstance(value, UUID):
                run_data[key] = str(value)
        runs_dict.append(run_data)

    return runs_df, runs_dict
    

if __name__ == "__main__":

    instances = load_swebench_dataset(name="princeton-nlp/SWE-bench_Verified")
    instance_ids = [instance["instance_id"] for instance in instances]
    
    
    for instance_id in tqdm(instance_ids):
        project_name = f"unresolved_{instance_id}"
        # if os.path.exists(f"{save_dir}/{project_name}"):
        #     continue
        try:
            runs_df, runs_dict = get_runs(project_name)
        except Exception as e:
            with open(f"{save_dir}/export.log", "a") as f:
                f.write(f"{project_name} has no runs\n")
            continue
        if len(runs_df) == 0:
            with open(f"{save_dir}/export.log", "a") as f:
                f.write(f"{project_name} has no runs\n")
            continue
        os.makedirs(f"{save_dir}/{instance_id}", exist_ok=True)

        with open(f"{save_dir}/{instance_id}/runs.json", "w") as f:
            json.dump(runs_dict, f, default=str)

        runs_df.to_excel(f"{save_dir}/{instance_id}/runs.xlsx")

