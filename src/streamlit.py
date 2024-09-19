import streamlit as st
import utils.config as load_config
import pandas as pd
import numpy as np
import eda.exploration as eda
import eda.feature as feature

st.title("Semantic Textual Similarity")

st.header("label 별 데이터 분포")
# st.write(dir(eda))
config = load_config.load_config()
data = pd.read_csv(config["path"]["train"])
st.pyplot(eda.vizLabel(data))
st.write("""
         - 데이터가 비어 있는 label이 거의 절반을 차지하고 있어서, 
            `bins` 값을 가능한 label의 개수의 절반인 26으로 설정하였습니다.
         - label이 0~0.1인 데이터가 가장 많고,
           유사도가 높은 데이터는 적은 것으로 보입니다.
         """)

st.header("토큰 길이 분포")
data_tokenized = feature.addTokenLength(data, config)
fig = eda.vizTokenLength(data)
def outlier(data, column):
    q1 = np.percentile(data[column], 25)
    q3 = np.percentile(data[column], 75)
    iqr = q3 - q1
    return data[(data[column] < q1 - 1.5 * iqr) | (data[column] > q3 + 1.5 * iqr)]
st.pyplot(fig)
st.write(f"""
         - {config["model"]["name"]}에서 사용한 토크나이저를 사용하였습니다.
         - 토큰 길이가 비상적으로 긴 데이터(70이상)은 각각 1개, 2개 존재합니다.
         - outlier의 개수는 각각 
         {len(outlier(data_tokenized, "tokenLength_1"))}개, 
         {len(outlier(data_tokenized, "tokenLength_2"))}개 입니다.
         """)