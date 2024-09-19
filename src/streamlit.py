import streamlit as st
import utils.config as load_config
import pandas as pd
import eda.exploration as eda

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
