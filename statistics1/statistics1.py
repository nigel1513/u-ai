# basic import
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# third party import
import pandas_profiling
import pygwalker as pyg
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
from st_aggrid import AgGrid, ColumnsAutoSizeMode
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 파일 업로드 함수화
@st.cache_data
def read_file(file):
    if  'csv' in file.name:
        df = pd.read_csv(file, encoding='UTF-8')
    elif 'xls' or 'xlsx' in file.name:
        df = pd.read_excel(file, engine='openpyxl')
    else:
        st.warning("형식에 맞는 데이터를 넣어주세요.")
        
    return df
# df.info() 함수
def df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True, null_counts=False)
    s = buffer.getvalue()
    st.text(s)



# 프로그램 시작 부분
def statistics_app():
    st.title("🔍 Descriptive statistics")

    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.subheader("1. 파일 업로드")
    st.write("\n")
    st.write("\n")
    st.write("\n")

    uploaded_files = st.file_uploader("CSV 파일을 업로드해주세요.", type=['xlsx', 'csv'])
    if uploaded_files is not None:    
        st.success('파일업로드 완료', icon="🔥")
        # 전체 데이터 확인하기
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.markdown("---")
        st.subheader("2. 전체 데이터 확인하기")
        df = read_file(uploaded_files)
        st.dataframe(df)
        
        st.markdown("---")
        st.subheader("3. Pygwalker를 사용한 시각화")
        pyg_html = pyg.walk(df, return_html=True)
        components.html(pyg_html, width=1000,height=900, scrolling=True) 

        # pandas profiling 사용 시 주석 제거
        st.markdown("---")
        st.subheader("4. Pandas Profiling으로 데이터 확인하기")
        st.info('컬럼별 자세한 내용은 Toggle Details를 눌러주세요', icon="🔥")
        pr = df.profile_report()
        st_profile_report(pr)

        st.write("\n")
        st.write("\n")
        st.write("\n")



    # 샘플데이터 업로드 부분
    else:
        sample_data = st.checkbox('샘플데이터 사용', value=True)
        if sample_data:
            st.warning('SAMPLE 데이터로 확인중', icon="⚠️")

            st.write("\n")
            st.write("\n")
            st.write("\n")

            # 전체 데이터 확인하기
            st.markdown("---")
            st.subheader("2. 전체 데이터 확인하기")
            df = pd.read_csv("sample_data/sample_df.csv")
            st.dataframe(df)

            st.write("\n")
            st.write("\n")
            st.write("\n")

            st.markdown("---")
            st.subheader("3. Pygwalker를 사용한 시각화")
            pyg_html = pyg.walk(df, return_html=True)
            components.html(pyg_html, width=1000,height=900, scrolling=True) 

            st.write("\n")
            st.write("\n")
            st.write("\n")

            st.markdown("---")
            # pandas profiling 사용 시 주석 제거
            st.subheader("4. Pandas Profiling으로 데이터 확인하기")
            st.info('컬럼별 자세한 내용은 Toggle Details를 눌러주세요', icon="🔥")
            pr = df.profile_report()
            st_profile_report(pr)             

        else:
            st.error("데이터를 넣어주세요.")
