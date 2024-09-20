from data_pipeline.augment_func.AugFunction import AugFunction
from utils.decorators import augment_func
import random

@augment_func("swap_sentences")
class swap_sentences(AugFunction):
    def __init__(self, **params):
        super().__init__()
        self.prob = params.get("probability", 0.0)
        self.skip_zero = params.get("skip_zero", True)
        self.text_columns = params.get("text_columns", ["sentence_1", "sentence_2"])
        
    def __call__(self, item):
        """
        증강한 데이터만 반환합니다. (원본 데이터는 반환하지 않습니다.)
        """
        if random.random() > self.prob or (self.skip_zero and item['label'] == 0):
            return self.empty_item(item)
        augmented_item = item.copy()
        augmented_item[self.text_columns[0]], augmented_item[self.text_columns[1]] = (
            augmented_item[self.text_columns[1]], augmented_item[self.text_columns[0]]
        )
        return self.merge_items([augmented_item])
        