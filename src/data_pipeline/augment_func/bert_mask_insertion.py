import importlib.util
import os
import random
from data_pipeline.augment_func.AugFunction import AugFunction
from utils.decorators import augment_func

@augment_func("bert_mask_insertion")
class BertMaskInsertion(AugFunction):
    def __init__(self, **params):
        super().__init__()
        self.ratio = params.get("ratio", 0.0)
        self.prob = params.get("probability", 1.0)
        self.label_stragegies = ["useConstant", "useRatio", "useModel", "useRandom"]
        self.label_strategy = params.get("label_strategy", "useConstant")
        assert self.label_strategy in self.label_stragegies, f"Label strategy must be one of {self.label_stragegies}"
        if self.label_strategy == "useConstant":
            self.constant = params.get("constant", 5.0)
        elif self.label_strategy == "useRatio":
            self.min = params.get("min", 0.0)
            self.max = params.get("max", 5.0)
        elif self.label_strategy == "useModel":
            self.model_path = params.get("model_path", "./model.pt")
        elif self.label_strategy == "useRandom":
            self.min = params.get("min", 0.0)
            self.max = params.get("max", 5.0)
        self.text_columns = params.get("text_columns", ["sentence_1", "sentence_2"])
        self.target_columns = params.get("target_columns", ["label"])
        self.bmi = self.importBERTAugmentation()
    
    def importBERTAugmentation(self):
        file_path = os.path.abspath("./K-TACC/BERT_augmentation.py")
        spec = importlib.util.spec_from_file_location("BERT_augmentation", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BERT_Augmentation()

    def makeLabel(self, target_col):
        label = 0
        if self.label_strategy == "useConstant":
            label = self.constant
        elif self.label_strategy == "useRandom":
            label = random.uniform(self.min, self.max)
        elif self.label_strategy == "useRatio":
            label = self.min + (self.max - self.min) * (1 - self.ratio)
        elif self.label_strategy == "useModel":
            raise NotImplementedError("Model-based label generation is not implemented yet.")
        label = round(label, 1)
        return label

    def __call__(self, item):
        if random.random() > self.prob:
            return self.empty_item(item)
        item_list = []
        for col in self.text_columns:
            auged = item.copy()
            counterpart = self.text_columns[0 if col == self.text_columns[1] else 1]
            auged[counterpart] = self.bmi.random_masking_insertion(item[col], self.ratio)
            for target_col in self.target_columns:
                auged[target_col] = self.makeLabel(target_col)
            item_list.append(auged)
        return self.merge_items(item_list)