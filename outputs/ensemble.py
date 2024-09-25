import os
import pandas as pd

# 폴더 경로 설정 (현재 폴더)
folder_path = os.path.dirname(os.path.abspath(__file__))

# 모든 csv 파일 불러오기
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 각 파일에 대한 가중치 설정 (예시로 각 파일에 대해 가중치를 설정)
weights = {
    'file1.csv': 1,
    'file2.csv': 1,
    'file3.csv': 1,
    'file4.csv': 1,
    'file5.csv': 1,
    # 추가 파일 및 가중치 설정
}

# 파일의 개수와 가중치가 일치하지 않을 때 에러메시지
if len(csv_files) != len(weights):
    raise ValueError("The number of CSV files does not match the number of weights provided.")

# 데이터프레임 리스트 초기화
dfs = []

# 각 csv 파일을 읽어서 데이터프레임 리스트에 추가
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    # 파일이 존재하지 않을 때 에러메시지
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file} does not exist.")
    df = pd.read_csv(file_path)
    df['weight'] = weights.get(file, 1)  # 가중치가 설정되지 않은 파일은 기본값 1로 설정
    dfs.append(df)

# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(dfs)

# 가중평균 계산
combined_df['weighted_target'] = combined_df['target'] * combined_df['weight']
result_df = combined_df.groupby('id').apply(
    lambda x: pd.Series({
        'target': (x['weighted_target'].sum() / x['weight'].sum()).round(1)
    })
).reset_index()

# 결과를 csv 파일로 저장
result_df.to_csv(os.path.join(folder_path, 'ensemble.csv'), index=False)