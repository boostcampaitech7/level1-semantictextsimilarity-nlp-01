from data_pipeline.augment_func.AugFunction import AugFunction
from utils.decorators import augment_func
import random

@augment_func("swap_sentences") # 데코레이터로 클래스의 이름을 설정해야 합니다. (config.yaml에서 사용되는 이름)
class swap_sentences(AugFunction): # 데이터 증강 함수는 AugFunction을 상속받기로 합시다.
    def __init__(self, **params):
        super().__init__()
        # config.yaml으로부터 받아올 parameter를 정의할 수 있습니다.
        self.prob = params.get("probability", 0.0) # "probability"라는 key가 없으면 0.0을 기본값으로 합니다.
        self.skip_zero = params.get("skip_zero", True)
        self.text_columns = params.get("text_columns", ["sentence_1", "sentence_2"])
        
    def __call__(self, item): # 증강한 데이터만 반환하도록 합시다.
        """
        증강한 데이터만 반환합니다. (원본 데이터는 반환하지 않습니다.)
        """
        if random.random() > self.prob or (self.skip_zero and item['label'] == 0):
            return self.empty_item(item) # 증강한 데이터가 없으면 empty_item을 반환합니다.
        augmented_item = item.copy()
        augmented_item[self.text_columns[0]], augmented_item[self.text_columns[1]] = (
            augmented_item[self.text_columns[1]], augmented_item[self.text_columns[0]]
        )
        return self.merge_items([augmented_item]) # 증강한 데이터를 반환합니다. merge하는 이유는 증강 후 데이터가 여러 개일 수 있기 때문입니다.
        