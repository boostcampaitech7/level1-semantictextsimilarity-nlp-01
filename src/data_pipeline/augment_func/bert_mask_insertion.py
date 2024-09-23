import importlib.util
import os
from data_pipeline.augment_func.AugFunction import AugFunction
from utils.decorators import augment_func

@augment_func("bert_mask_insertion")
class BertMaskInsertion(AugFunction):
    def __init__(self, **params):
        super().__init__()
        self.ratio = params.get("ratio", 0.0)
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
        self.bmi = self.importModule()
    
    def importModule(self):
        file_path = os.path.abspath("../K-TACC/BERT_augmentation.py")
        spec = importlib.util.spec_from_file_location("BERT_augmentation", file_path)
        return importlib.util.module_from_spec(spec)

    def __call__(self, item):
        return self.empty_item(item)