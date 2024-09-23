import os
import pandas as pd
import importlib

from tqdm.auto import tqdm

class UnderSampling():
    def __init__(self, config):
        self.under_sampling = config["data"].get("under_sampling", [])
        self.mappings = self.init_mapping()
        
    def init_mapping(self):
        pwd = os.path.dirname(__file__)
        us_dir = os.path.join(pwd, "under_sampling_func")
        us_files = os.listdir(us_dir)
        dict = {}
        for us_file in us_files:
            if us_file.endswith(".py"):
                module_name = us_file[:-3]
                module = importlib.import_module(f"data_pipeline.under_sampling_func.{module_name}")
                for name in dir(module):
                    obj = getattr(module, name)
                    if hasattr(obj, "is_under_sampling_func") and hasattr(obj, "call_name"):
                        dict[obj.call_name] = obj
        return dict
    
    def __call__(self, data):
        under_sampling_data = []
        for idx, item in tqdm(data.iterrows(), desc='under_sampling', total=len(data)):
            for us in self.under_sampling:
                map_func = self.mappings[us["method"]](**us["params"])
                if us["method"] in self.mappings:
                    under_sampled = map_func(item)
                    if under_sampled.empty:
                        continue
                    under_sampling_data.append(under_sampled)
        under_sampling_data.append(data)
        rtn = pd.concat(under_sampling_data, ignore_index=True)  # 샘플링된 데이터를 반환
        return rtn