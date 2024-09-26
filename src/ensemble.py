import os
import pandas as pd
import argparse

def auto_ensemble(output_path='outputs', result_file='ensemble.csv'):
    # 폴더 경로 설정 (현재 폴더)
    folder_path = os.path.dirname(os.path.join(os.getcwd(), output_path+'/'))

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
    result_df.to_csv(os.path.join(folder_path, result_file), index=False)
    print(f'현재 디렉토리에 다음 파일이 저장되었습니다: {result_file}')
    
def interactive_ensemble(output_path='outputs', result_file='ensemble.csv'):
    # 폴더 경로 설정 (현재 폴더)
    folder_path = os.path.dirname(os.path.join(os.getcwd(), output_path+'/'))

    # 모든 csv 파일 불러오기
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    weight_df = pd.DataFrame(columns=['name', 'weight'])
    for csv in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv))
        df['name'] = csv
        weight_df = pd.concat([weight_df, pd.DataFrame({'name': [csv], 'weight': [0]})])
        dfs.append(df)
    
    csv_df = pd.concat(dfs)
    page = 0
    while True:
        print(f"Page {page + 1}")
        for i, file in enumerate(csv_files[page * 10:page * 10 + 10]):
            weight = weight_df.loc[weight_df['name'] == file, 'weight'].values[0]
            print(f"[{i}] [{'v' if weight != 0 else '-'}] {file} {'(w=' + str(weight) + ')'if weight != 0 else ''}")
        x = input("csv파일을 선택하세요. [0-9,a,s,d] ")
        if x == 'a':
            page = max(0, page - 1)
        elif x == 'd':
            page = min(len(csv_files) // 10, page + 1)
        elif x == 's':
            x = input("종료하시겠습니까? [y/n] ")
            if x == 'y':
                break
            else:
                continue
        else:
            try:
                idx = int(x)
                if idx + page * 10 < len(csv_files):
                    selected_file = csv_files[idx + page * 10]
                    print(f"Selected file: {selected_file}")
                    weight = input("Enter the weight for this file: ")
                    weight_df.loc[weight_df['name'] == selected_file, 'weight'] = float(weight)
            except:
                print("Invalid input. Please select a number between 0 and 9 or press 's' to stop.")
        print("현재 선택된 파일들의 가중치")
        print(weight_df[weight_df['weight'] != 0])
    
    merged_df = pd.merge(csv_df, weight_df, left_on='name', right_on='name')
    merged_df['weighted_target'] = merged_df['target'] * merged_df['weight']
    if merged_df['weight'].sum() == 0:
        print("가중치가 모두 0입니다. 가중평균을 계산할 수 없습니다.")
        return
    result_df = merged_df.groupby('id').apply(
        lambda x: pd.Series({
            'target': round(x['weighted_target'].sum() / x['weight'].sum(), 1)
        })
    ).reset_index()
    
    result_df.to_csv(result_file, index=False)
    print(f'현재 디렉토리에 다음 파일이 저장되었습니다: {result_file}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='interactive', help='interactive or auto mode')
    parser.add_argument('--outputs', type=str, default ='outputs', help='path to the output folder')
    parser.add_argument('--result', type=str, default='ensemble.csv', help='name of the ensemble result file')
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_ensemble(args.outputs, args.result)
    elif args.mode == 'auto':
        auto_ensemble(args.outputs, args.result)
    else:
        raise ValueError("Invalid mode. Please select either 'interactive' or 'auto'.")
    