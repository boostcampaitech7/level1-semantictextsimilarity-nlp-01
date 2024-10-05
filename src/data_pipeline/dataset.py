import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets
        self.is_inference = len(targets) == 0 # 타겟 데이터가 비어있으면 inference 모드로 설정합니다
        
        # 디버깅
        print(f"Dataset initialized with {len(self.inputs)} samples")
        if len(self.inputs) > 0:
            print(f"Sample input: {self.inputs[0]}")
            
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        
        # 기존에 attention_mask를 사용하지 않았던 것에서 벗어나
        # 해당 정보를 살리기 위해 코드 추가
        input_data = {
            'input_ids': torch.tensor(self.inputs[idx]['input_ids']),
            'attention_mask': torch.tensor(self.inputs[idx]['attention_mask'])
        }
        
        
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if self.is_inference:
            return input_data
        else:
            return input_data, torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)
