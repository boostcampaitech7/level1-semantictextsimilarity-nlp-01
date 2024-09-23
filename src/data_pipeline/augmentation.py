import os
import pandas as pd
import importlib

from tqdm.auto import tqdm

class Augmentation():
    def __init__(self, config):
        self.augmentation = config["data"].get("augmentation", [])
        self.mappings = self.init_mapping()
        
    def init_mapping(self):
        pwd = os.path.dirname(__file__)
        aug_dir = os.path.join(pwd, "augment_func")
        aug_files = os.listdir(aug_dir)
        dict = {}
        for aug_file in aug_files:
            if aug_file.endswith(".py"):
                module_name = aug_file[:-3]
                module = importlib.import_module(f"data_pipeline.augment_func.{module_name}")
                for name in dir(module):
                    obj = getattr(module, name)
                    if hasattr(obj, "is_augment_func") and hasattr(obj, "call_name"):
                        dict[obj.call_name] = obj
        return dict
    
    def __call__(self, data):
        augmented_data = []
        sampled_data = []
        for idx, item in tqdm(data.iterrows(), desc='augmenting', total=len(data)):
            for aug in self.augmentation:
                map_func = self.mappings[aug["method"]](**aug["params"])
                if aug["method"] in self.mappings:
                    augmented = map_func(item)
                    if augmented.empty:
                        continue
                    if aug["method"] == 'undersample_label_0':
                        sampled_data.append(augmented)
                    else:
                        augmented_data.append(augmented)
        augmented_data.append(data)
        augmented_data_df = pd.concat(augmented_data)
        sampled_data_df = pd.concat(sampled_data)
        tmp = pd.concat([augmented_data_df, sampled_data_df]).drop_duplicates(keep=False)
        sampled_data_df['sentence_2'] = sampled_data_df['sentence_1']
        sampled_data_df['label'] = 5.0
        sampled_data_df['binary-label'] = 1.0

        rtn = pd.concat([tmp, sampled_data_df]).drop_duplicates(keep=False)
        # rtn = pd.concat(augmented_data)
        return rtn