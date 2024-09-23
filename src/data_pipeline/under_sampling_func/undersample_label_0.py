from data_pipeline.under_sampling_func.UnderSamplingFunction import UnderSamplingFunction
from utils.decorators import under_sampling_func
import random
import pandas as pd

@under_sampling_func("undersample_label_0") # 데코레이터로 클래스의 이름을 설정해야 합니다. (config.yaml에서 사용되는 이름)
class undersample_label_0(UnderSamplingFunction): # 데이터 샘플링 함수는 UnderSamplingFunction 상속받기로 합시다.
    def __init__(self, **params):
        super().__init__()
        # config.yaml으로부터 받아올 parameter를 정의할 수 있습니다.
        self.prob = params.get("probability", 1.0)  # 언더샘플링 확률 (기본값은 1.0, 즉 모든 0 라벨을 대상으로 함)
        self.target_label = 0  # 언더샘플링 대상 라벨 (여기서는 라벨이 0인 경우)
        
    def __call__(self, item): # 샘플링한 데이터만 반환하도록 합시다.
        """
        라벨이 0인 데이터를 확률에 따라 언더샘플링하여 반환합니다.
        """
        # 라벨이 0인 경우 확률적으로 제거
        if item['label'] == self.target_label and random.random() > self.prob:
            return self.empty_item(item)  # 라벨이 0이고 확률을 넘지 못하면 제거
        return self.merge_items([item])  # 나머지 데이터는 그대로 유지
        


