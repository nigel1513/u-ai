import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import sklearn
import base64 # Standard Python Module 
from io import StringIO, BytesIO # Standard Python Module
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


@st.cache_data
def load_dataframe(upload_file):
    try:
        df = pd.read_csv(upload_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(upload_file)

    columns = list(df.columns)
    columns.insert(0, None)
    return df, columns 

# 타겟 데이터 업로드
def target_dataframe(upload_target):
    global target
    try:
        target = pd.read_csv(upload_target)
        
    except Exception as e:
        print(e)
        target = pd.read_excel(upload_target)

    return target 

# 4. drop columns

def drop_df(df):
    global features
    st.write('\n')
    st.write('\n')
    select_dropOrnot = st.selectbox("컬럼을 삭제하시겠습니까?",("No", "Yes"))
    if select_dropOrnot == "Yes":
        select_drop_columns = st.multiselect("삭제할 컬럼을 고르세요", df.columns)
        if len(select_drop_columns) >= 1 :
            features = df.drop(axis=1, columns=select_drop_columns)
      
            st.dataframe(features)
            st.success(str(select_drop_columns) + " " + "컬럼 삭제가 완료되었습니다", icon="✅")
            return features
        elif len(select_drop_columns) ==0 :
            features = df
            st.write('\n')
            st.write('\n')
            st.dataframe(features)
            st.warning("삭제된 컬럼이 없습니다", icon="⚠️")
            return features
    else:
        features = df
        st.dataframe(features)
        st.warning("삭제된 컬럼이 없습니다", icon="⚠️")
        return features

# Drop_na
def Drop_na(df):
    global features
    drop_columns = st.selectbox("빈값을 가진 컬럼을 삭제하시겠습니까?",('No', 'Yes'))
    if drop_columns == "Yes":
        if features.isnull().sum().sum() == 0:
            st.dataframe(features)
            st.success("Null 값을 가진 컬럼이 없습니다.")
            return features
        else:
            drop_method = st.selectbox("삭제방법",('any', 'all'), help='any= 빈값이 하나라도 있는 경우' +" "+ " " + 'all= 전체가 빈값일 경우')
            if drop_method is not None:
                try:
                    drop_axis = st.selectbox("삭제할 축을 선택해주세요.", (0, 1), help='0 = 행기준 삭제' +" "+ " " +  '1 = 열기준 삭제')
                    features = df.dropna(how=drop_method, axis=drop_axis)
                    st.write('\n')
                    st.write('\n')
                    st.dataframe(features)
                    feautures_null_columns = features.isnull().sum().reset_index()
                    feautures_null_columns = feautures_null_columns.rename(columns={'index':'index', 0:'null'})
                    feautures_null_columns_list = feautures_null_columns[feautures_null_columns['null'] > 0]['index'].to_list()

                    if len(feautures_null_columns_list) == 0:
                        st.success('빈값 컬럼 삭제를 완료하였습니다.', icon="✅")
                    
                    else:
                        feautures_null_columns = str(feautures_null_columns[feautures_null_columns['null'] > 0]['index'].to_list())
                        st.error(feautures_null_columns + " " + "컬럼에 빈값이 존재합니다.")

                    return features
                except Exception as e:
                    print(e)
    else:
        st.dataframe(features)
        feautures_null_columns = features.isnull().sum().reset_index()
        feautures_null_columns = feautures_null_columns.rename(columns={'index':'index', 0:'null'})
        feautures_null_columns_list = feautures_null_columns[feautures_null_columns['null'] > 0]['index'].to_list()

        if len(feautures_null_columns_list) == 0:
            st.success("빈값을 가진 컬럼은 없습니다.", icon="✅")
        else:
            feautures_null_columns = str(feautures_null_columns[feautures_null_columns['null'] > 0]['index'].to_list())
            st.error(feautures_null_columns + " " + "컬럼에 빈값이 존재합니다.")

        return features

# 3. 특수문자 제거
def special_str_drop(features):
    special_drop = st.multiselect("특수문자를 제거할 컬럼을 선택하세요.", features.columns.to_list(), default=None, key="A0", help="+, -, .을 제외한 특수문자 제거")

    if len(special_drop) == 0:
        st.dataframe(features)
        st.info("특수문자가 제거된 컬럼은 없습니다.")


    elif len(special_drop) > 0:
        features[special_drop] = features[special_drop].replace(to_replace=r'[^\w\.\+\-]', value=r'', regex=True)
        st.dataframe(features)
        st.success( str(special_drop) +" " + " 컬럼 특수문자 제거가 완료되었습니다.", icon="✅")



    else:
        st.write("error")

    return features

# 4. train_test_split
def split_train_test_split(features):
    val = None
    validation_select = st.selectbox("검증데이터를 분리하시겠습니까?",('No', 'Yes'), help="검증데이터를 사용할 경우(Train, Test, val) 3개로 나누어짐")
    stratify_select = st.selectbox("계층형 분리를 사용하시겠습니까?", ('No', 'Yes'), help="계층분리할 컬럼이 Null 값이 있거나 데이터 분류값이 하나일 경우에는 사용 불가능")
    
    if validation_select == 'Yes':
        if stratify_select == 'No':
            df = features
            df_columns = df.columns.to_list()

            test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요", min_value=0.1, max_value=0.9, format='%.2f')
            val_size_input = st.slider("검증 데이터 사이즈 비율을 선택하세요", min_value=0.1, max_value=0.9, format='%.2f')
            train, test = train_test_split(df, test_size=test_size_input, random_state=42)       
            train, val = train_test_split(train, test_size=val_size_input, random_state=42)
            st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Size", len(df))
            col2.metric("Train Size", len(train))
            col3.metric("Test Size", len(test))
            col4.metric("Validation Size", len(val))
        
        else:
            df = features
            df_columns = df.columns.to_list()

            stratify_columns = st.multiselect("계층분리할 컬럼을 선택하세요.", df_columns)
            stratify_columns_count = len(stratify_columns)
            if stratify_columns_count == 1:
                try:
                    test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요.", min_value=0.1, max_value=0.9, format='%.2f')
                    val_size_input = st.slider("검증 데이터 사이즈 비율을 선택하세요", min_value=0.1, max_value=0.9, format='%.2f')
                    train, test = train_test_split(df, test_size=test_size_input, stratify=df[stratify_columns], random_state=42)
                    train, val = train_test_split(train, test_size=val_size_input, stratify=train[stratify_columns],random_state=42)
                    # value_counst() 확인한 결과 2번 나눠서 그런지 일부 비율이 안맞는데 맞출 수 있는 방법 고민해봐야함.

                    st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Size", len(df))
                    col2.metric("Train Size", len(train))                
                    col3.metric("Test Size", len(test))
                    col4.metric("Validation Size", len(val))
                except Exception as e:
                    st.error("계층분리 컬럼에 Null 값이 있거나 계층분리가 불가능한 상태입니다.")

            elif stratify_columns_count == 0:
                st.info("계층 분리할 컬럼을 선택해주세요.")
                
           
            elif stratify_columns_count >= 2:
                features['multi_columns_Stratify'] = ""
                for i in stratify_columns:
                    df['multi_columns_Stratify'] = df['multi_columns_Stratify'] + "_" +df[i].astype(str)            
                    
                test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요.", min_value=0.1, max_value=0.9, format='%.2f')
                val_size_input = st.slider("검증 데이터 사이즈 비율을 선택하세요", min_value=0.1, max_value=0.9, format='%.2f')
                try:
                    train, test = train_test_split(df, test_size=test_size_input, stratify=df['multi_columns_Stratify'], random_state=42)
                    train, val = train_test_split(train, test_size=val_size_input, stratify=train['multi_columns_Stratify'],random_state=42)
                except Exception as e:
                    st.error("계층분리 컬럼에 Null 값이 있거나 계층분리가 불가능한 상태입니다.")
                                  
                st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Size", len(df))
                col2.metric("Train Size", len(train))
                col3.metric("Test Size", len(test))
                col4.metric("Validation Size", len(val))             
                
            else:   
                st.error("에러가 발생하였습니다.")
       
    else:
        if stratify_select == 'No':
            df = features
            df_columns = df.columns.to_list()

            test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요", min_value=0.1, max_value=0.9, format='%.2f')
            train, test = train_test_split(df, test_size=test_size_input, random_state=42)       
            st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
            col1, col2, col3  = st.columns(3)
            col1.metric("Total Size", len(df))
            col2.metric("Train Size", len(train))
            col3.metric("Test Size", len(test))
        
        else:
            df = features
            df_columns = df.columns.to_list()
            stratify_columns = st.multiselect("계층분리할 컬럼을 선택하세요.", df_columns)
            stratify_columns_count = len(stratify_columns)
            if stratify_columns_count == 1:
                try:
                    test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요.", min_value=0.1, max_value=0.9, format='%.2f')
                    train, test = train_test_split(df, test_size=test_size_input, stratify=df[stratify_columns], random_state=42)
                    # value_counst() 확인한 결과 2번 나눠서 그런지 일부 비율이 안맞는데 맞출 수 있는 방법 고민해봐야함.

                    st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Size", len(df))
                    col2.metric("Train Size", len(train))                
                    col3.metric("Test Size", len(test))
                except Exception as e:
                    st.error("계층분리 컬럼에 Null 값이 있거나 계층분리가 불가능한 상태입니다.")

            elif stratify_columns_count >= 2:
                df['multi_columns_Stratify'] = ""
                for i in stratify_columns:
                    df['multi_columns_Stratify'] = df['multi_columns_Stratify'] + "_" +df[i].astype(str)               
                test_size_input = st.slider("테스트 데이터 사이즈 비율을 선택하세요.", min_value=0.1, max_value=0.9, format='%.2f')
                try:
                    train, test = train_test_split(df, test_size=test_size_input, stratify=df['multi_columns_Stratify'], random_state=42)
                    # value_counst() 확인한 결과 2번 나눠서 그런지 일부 비율이 안맞는데 맞출 수 있는 방법 고민해봐야함.
                except Exception as e:
                    st.error("계층분리 컬럼에 Null 값이 있거나 계층분리가 불가능한 상태입니다.")
                                
                st.info("분리된 데이터 사이즈를 체크하세요.", icon="ℹ️")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Size", len(df))
                col2.metric("Train Size", len(train))
                col3.metric("Test Size", len(test))
    return train, test, val


# Fill_Na(only numeric)
def fill_na(train, test, val):
    if val is not None:
        train_df_numeric = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
        test_df_numeric = test.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
        val_df_numeric = val.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()




        df_object = train.select_dtypes(include = 'object').columns.to_list()
        df_datetime = train.select_dtypes(include = 'datetime').columns.to_list()
        train_isnull = train.isnull().sum().sum()
        test_isnull = test.isnull().sum().sum()
        val_isnull = val.isnull().sum().sum()

        fill_columns = st.selectbox("빈컬럼을 채우시겠습니까?",('No', 'Yes'))


        if fill_columns == 'No':
            if (train_isnull == 0) & (test_isnull== 0) & (val_isnull==0):
                st.success('train, test, val 데이터에 빈컬럼이 없습니다.', icon="✅")
                
            else:                
                data_dtypes = train.dtypes.reset_index()
                data_dtypes = data_dtypes.rename(columns = {'index':'Column 명', 0:'dtypes'})
                train_isnull_columns = train.isnull().sum().reset_index()
                train_isnull_columns = train_isnull_columns.rename(columns = {'index':'Column 명', 0:'train'})
                test_isnull_columns = test.isnull().sum().reset_index()
                test_isnull_columns = test_isnull_columns.rename(columns = {'index':'Column 명', 0:'test'})
                val_isnull_columns = val.isnull().sum().reset_index()
                val_isnull_columns = val_isnull_columns.rename(columns = {'index':'Column 명', 0:'val'})
                train_isnull_columns['test'] = test_isnull_columns['test']
                train_isnull_columns['val'] = val_isnull_columns['val']
                train_isnull_columns['dtypes'] = data_dtypes['dtypes']             
                train_isnull_columns = train_isnull_columns[['Column 명', 'dtypes','train','test', 'val']]        
                st.write('\n')
                st.markdown("**:blue[6-1. 훈련, 테스트, 검증 데이터셋 Null 개수 확인]**")
                st.dataframe(train_isnull_columns)
                st.error("빈값을 가진 컬럼이 있어 값을 채워야 합니다.", icon="🚨")

        else:
            if (train_isnull != 0) or (test_isnull != 0) or (val_isnull !=0):
                data_dtypes = train.dtypes.reset_index()
                data_dtypes = data_dtypes.rename(columns = {'index':'Column 명', 0:'dtypes'})
                train_isnull_columns = train.isnull().sum().reset_index()
                train_isnull_columns = train_isnull_columns.rename(columns = {'index':'Column 명', 0:'train'})
                test_isnull_columns = test.isnull().sum().reset_index()
                test_isnull_columns = test_isnull_columns.rename(columns = {'index':'Column 명', 0:'test'})
                val_isnull_columns = val.isnull().sum().reset_index()
                val_isnull_columns = val_isnull_columns.rename(columns = {'index':'Column 명', 0:'val'})
                train_isnull_columns['test'] = test_isnull_columns['test']
                train_isnull_columns['val'] = val_isnull_columns['val']
                train_isnull_columns['dtypes'] = data_dtypes['dtypes']             
                train_isnull_columns = train_isnull_columns[['Column 명', 'dtypes','train','test', 'val']]       


                st.markdown('')
                st.markdown("**:blue[6-1. 훈련, 테스트, 검증 데이터셋 Null 개수 확인]**")
                st.dataframe(train_isnull_columns)

                st.markdown('')
                st.markdown("**:blue[6-2. 수치형 데이터 채우기]**")
                groupOrNongroup = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True)
                if groupOrNongroup == "Column":
                    fill_na_columns = st.selectbox("어떤값으로 채우시겠습니까?",(0, 'mean','min','max','median'))
                    st.markdown('')
                    if fill_na_columns == 'mean':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Mean 값]**")
                        st.dataframe(train[train_df_numeric].mean().reset_index().rename(columns = {'index':'Columns', 0:'Columns Mean Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].mean())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].mean())
                        val[val_df_numeric] = val[val_df_numeric].fillna(train[train_df_numeric].mean())
                                              

                        
                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-4. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)


                    elif fill_na_columns == 'min':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Min 값]**")
                        st.dataframe(train[train_df_numeric].min().reset_index().rename(columns = {'index':'Columns', 0:'Columns Min Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].min())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].min())
                        val[val_df_numeric] = val[val_df_numeric].fillna(train[train_df_numeric].min())


                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)


                    elif fill_na_columns == 'max':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Max 값]**")
                        st.dataframe(train[train_df_numeric].max().reset_index().rename(columns = {'index':'Columns', 0:'Columns Max Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].max())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].max())
                        val[val_df_numeric] = val[val_df_numeric].fillna(train[train_df_numeric].max())
                        
                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)

                    elif fill_na_columns == 'median':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Median 값]**")
                        st.dataframe(train[train_df_numeric].median().reset_index().rename(columns = {'index':'Columns', 0:'Columns Median Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].median())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].median())
                        val[val_df_numeric] = val[val_df_numeric].fillna(train[train_df_numeric].median())

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)

                


                    elif fill_na_columns == 0:
                        train[train_df_numeric] = train[train_df_numeric].fillna(0)
                        test[test_df_numeric] = test[test_df_numeric].fillna(0)
                        val[val_df_numeric] = val[val_df_numeric].fillna(0)
                        st.success("수치형컬럼 Null값이 0으로 채워졌습니다.")

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)
                        
             
                    else:
                        pass



                elif groupOrNongroup == "Group":

                    fill_na_columns = st.selectbox("어떤값으로 채우시겠습니까?",('mean','min','max','median'))
                    groupby_columns = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                    
     
                    if fill_na_columns == 'mean':
                        
                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_mean' for x in numeric_only_columns]
                        

                        train_mean = train.groupby(groupby_columns)[numeric_only_columns].mean().reset_index()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 평균 데이터 값(수치형 데이터만)]**")
                        st.write(train_mean)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_mean, on=groupby_columns, how='left', suffixes=('', '_mean'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        for i in numeric_only_columns:
                            a = i + '_mean'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_mean, on=groupby_columns, how='left', suffixes=('', '_mean'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_mean'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                        val_update = val.merge(train_mean, on=groupby_columns, how='left', suffixes=('', '_mean'))
                        val_update = val_update.set_index(val.index)
                        val = val_update 
                        for i in numeric_only_columns:
                            a = i + '_mean'
                            val[i] = val[i].fillna(val[a])
                            val = val.drop(columns=[a])

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)
                    
                    elif fill_na_columns == 'min':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_min' for x in numeric_only_columns]

                        train_min = train.groupby(groupby_columns)[numeric_only_columns].min()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_min)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_min, on=groupby_columns, how='left', suffixes=('', '_min'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_min'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_min, on=groupby_columns, how='left', suffixes=('', '_min'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_min'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                        val_update = val.merge(train_min, on=groupby_columns, how='left', suffixes=('', '_min'))
                        val_update = val_update.set_index(val.index)
                        val = val_update 
                        for i in numeric_only_columns:
                            a = i + '_min'
                            val[i] = val[i].fillna(val[a])
                            val = val.drop(columns=[a])

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)

                    elif fill_na_columns == 'max':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_max' for x in numeric_only_columns]

                        train_max = train.groupby(groupby_columns)[numeric_only_columns].max()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_max)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_max, on=groupby_columns, how='left', suffixes=('', '_max'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_max'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_max, on=groupby_columns, how='left', suffixes=('', '_max'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_max'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                        val_update = val.merge(train_max, on=groupby_columns, how='left', suffixes=('', '_max'))
                        val_update = val_update.set_index(val.index)
                        val = val_update 
                        for i in numeric_only_columns:
                            a = i + '_max'
                            val[i] = val[i].fillna(val[a])
                            val = val.drop(columns=[a])

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)

                    elif fill_na_columns == 'median':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]
                        numeric_only_columns_update = [x + '_median' for x in numeric_only_columns]

                        train_median = train.groupby(groupby_columns)[numeric_only_columns].median()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_median)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_median, on=groupby_columns, how='left', suffixes=('', '_median'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_median'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_median, on=groupby_columns, how='left', suffixes=('', '_median'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_median'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                        val_update = val.merge(train_median, on=groupby_columns, how='left', suffixes=('', '_median'))
                        val_update = val_update.set_index(val.index)
                        val = val_update 
                        for i in numeric_only_columns:
                            a = i + '_median'
                            val[i] = val[i].fillna(val[a])
                            val = val.drop(columns=[a])

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)
                                val[object_columns] = val[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)
                                val[object_columns] = val[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                            val_update = val.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            val_update = val_update.set_index(val.index)
                            val = val_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                val[i] = val[i].fillna(val[a])
                                val = val.drop(columns=[a])
                    
                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        st.write('\n')
                        st.markdown("**:black[③Val]**")
                        st.dataframe(val)


                      
                else:
                    pass
                    
    
            else:   
                st.subheader("Fill Only Numeric Columns")
                st.success('There is not any NA value in your dataset.', icon="✅")


    else:
        train_df_numeric = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
        test_df_numeric = test.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
        df_object = train.select_dtypes(include = 'object').columns.to_list()
        df_datetime = train.select_dtypes(include = 'datetime').columns.to_list()
        train_isnull = train.isnull().sum().sum()
        test_isnull = test.isnull().sum().sum()

        fill_columns = st.selectbox("빈컬럼을 채우시겠습니까?",('No', 'Yes'))


        if fill_columns == 'No':
            if (train_isnull == 0) & (test_isnull== 0):
                st.success('train, test 데이터에 빈컬럼이 없습니다.', icon="✅")
                
            else:                
                data_dtypes = train.dtypes.reset_index()
                data_dtypes = data_dtypes.rename(columns = {'index':'Column 명', 0:'dtypes'})
                train_isnull_columns = train.isnull().sum().reset_index()
                train_isnull_columns = train_isnull_columns.rename(columns = {'index':'Column 명', 0:'train'})
                test_isnull_columns = test.isnull().sum().reset_index()
                test_isnull_columns = test_isnull_columns.rename(columns = {'index':'Column 명', 0:'test'})
                train_isnull_columns['test'] = test_isnull_columns['test']
                train_isnull_columns['dtypes'] = data_dtypes['dtypes']             
                train_isnull_columns = train_isnull_columns[['Column 명', 'dtypes','train','test']]        
                st.write('\n')
                st.markdown("**:blue[6-1. 훈련, 테스트, 검증 데이터셋 Null 개수 확인]**")
                st.dataframe(train_isnull_columns)
                st.error("빈값을 가진 컬럼이 있어 값을 채워야 합니다.", icon="🚨")


        else:
            if (train_isnull != 0) or (test_isnull != 0):
                data_dtypes = train.dtypes.reset_index()
                data_dtypes = data_dtypes.rename(columns = {'index':'Column 명', 0:'dtypes'})

                train_isnull_columns = train.isnull().sum().reset_index()
                train_isnull_columns = train_isnull_columns.rename(columns = {'index':'Column 명', 0:'train'})
                test_isnull_columns = test.isnull().sum().reset_index()
                test_isnull_columns = test_isnull_columns.rename(columns = {'index':'Column 명', 0:'test'})
                
                train_isnull_columns['test'] = test_isnull_columns['test']
                train_isnull_columns['dtypes'] = data_dtypes['dtypes']             
                train_isnull_columns = train_isnull_columns[['Column 명', 'dtypes','train','test']]       


                st.markdown('')
                st.markdown("**:blue[6-1. 훈련, 테스트, 검증 데이터셋 Null 개수 확인]**")
                st.dataframe(train_isnull_columns)

                st.markdown('')
                st.markdown("**:blue[6-2. 수치형 데이터 채우기]**")
                groupOrNongroup = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True)

                if groupOrNongroup == "Column":
                    fill_na_columns = st.selectbox("어떤값으로 채우시겠습니까?",(0, 'mean','min','max','median'))
                    st.markdown('')

                    if fill_na_columns == 'mean':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Mean 값]**")
                        st.dataframe(train[train_df_numeric].mean().reset_index().rename(columns = {'index':'Columns', 0:'Columns Mean Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].mean())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].mean())                                              

                        
                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')
                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

           
                    


                    elif fill_na_columns == 'min':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Min 값]**")
                        st.dataframe(train[train_df_numeric].min().reset_index().rename(columns = {'index':'Columns', 0:'Columns Min Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].min())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].min())


                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")#단일값만 있을 경우 값이 채워지지 않습니다.
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                    

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)



                    elif fill_na_columns == 'max':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Max 값]**")
                        st.dataframe(train[train_df_numeric].max().reset_index().rename(columns = {'index':'Columns', 0:'Columns Max Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].max())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].max())
                        
                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)




                    elif fill_na_columns == 'median':
                        st.markdown("**:blue[6-2-1. 훈련데이터 Column Median 값]**")
                        st.dataframe(train[train_df_numeric].median().reset_index().rename(columns = {'index':'Columns', 0:'Columns Median Values'}))

                        train[train_df_numeric] = train[train_df_numeric].fillna(train[train_df_numeric].median())
                        test[test_df_numeric] = test[test_df_numeric].fillna(train[train_df_numeric].median())

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])
                        
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)


                    elif fill_na_columns == 0:
                        train[train_df_numeric] = train[train_df_numeric].fillna(0)
                        test[test_df_numeric] = test[test_df_numeric].fillna(0)
                        st.success("수치형컬럼 Null값이 0으로 채워졌습니다.")

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup2 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup2>')


                        if groupOrNongroup2 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns2 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns2]
                           


                            train_mode = train.groupby(groupby_columns2)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns2, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')
    
                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                        

                           


  
                    



                    else:
                        pass
                
                elif groupOrNongroup == "Group":

                    fill_na_columns = st.selectbox("어떤값으로 채우시겠습니까?",('mean','min','max','median'))
                    groupby_columns = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.")
                    
     
                    if fill_na_columns == 'mean':
                        
                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_mean' for x in numeric_only_columns]
                        

                        train_mean = train.groupby(groupby_columns)[numeric_only_columns].mean().reset_index()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 평균 데이터 값(수치형 데이터만)]**")
                        st.write(train_mean)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_mean, on=groupby_columns, how='left', suffixes=('', '_mean'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        for i in numeric_only_columns:
                            a = i + '_mean'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_mean, on=groupby_columns, how='left', suffixes=('', '_mean'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_mean'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                    

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                    
                    elif fill_na_columns == 'min':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_min' for x in numeric_only_columns]

                        train_min = train.groupby(groupby_columns)[numeric_only_columns].min()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_min)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_min, on=groupby_columns, how='left', suffixes=('', '_min'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_min'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_min, on=groupby_columns, how='left', suffixes=('', '_min'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_min'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                    

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                        

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)

                       
                    elif fill_na_columns == 'max':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]

                        numeric_only_columns_update = [x + '_max' for x in numeric_only_columns]

                        train_max = train.groupby(groupby_columns)[numeric_only_columns].max()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_max)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_max, on=groupby_columns, how='left', suffixes=('', '_max'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_max'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_max, on=groupby_columns, how='left', suffixes=('', '_max'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_max'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           

                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)
                    

                    elif fill_na_columns == 'median':

                        st.write('\n')
                        numeric_only_columns = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
                        numeric_only_columns = [i for i in numeric_only_columns if i not in groupby_columns]
                        numeric_only_columns_update = [x + '_median' for x in numeric_only_columns]

                        train_median = train.groupby(groupby_columns)[numeric_only_columns].median()

                        st.markdown("**:blue[6-2-1. 훈련데이터 기준 그룹별 최소 데이터 값(수치형 데이터만)]**")
                        st.write(train_median)
                        st.info("훈련데이터 그룹별 데이터로 Null값 처리")

                        train_update = train.merge(train_median, on=groupby_columns, how='left', suffixes=('', '_median'))
                        train_update = train_update.set_index(train.index)
                        train = train_update
                        
                        for i in numeric_only_columns:
                            a = i + '_median'
                            train[i] = train[i].fillna(train[a])
                            train = train.drop(columns=[a])      
                                           
                        
                        test_update = test.merge(train_median, on=groupby_columns, how='left', suffixes=('', '_median'))  
                        test_update = test_update.set_index(test.index)
                        test = test_update    
                        for i in numeric_only_columns:
                            a = i + '_median'
                            test[i] = test[i].fillna(test[a])
                            test = test.drop(columns=[a])


                     

                        object_columns = train.select_dtypes(include=object).columns.to_list()
                        st.write('\n')
                        st.markdown("**:blue[6-3. 문자형 데이터 채우기]**")
                        groupOrNongroup3 = st.radio("Null 값 찾는 방법", ('Column', 'Group'), horizontal=True, key='<groupOrNongroup3>')


                        if groupOrNongroup3 == 'Column':
                            fill_na_object_columns = st.selectbox("How to FillNA",(0, 'mode'))
                            
                            if fill_na_object_columns == 0:
                                train[object_columns] = train[object_columns].fillna(0)
                                test[object_columns] = test[object_columns].fillna(0)

                            elif fill_na_object_columns == 'mode':

                                df_mode = train[object_columns].mode().transpose()                                
                                df_mode = df_mode.iloc[:,:1].squeeze()

                                st.write('\n')
                                st.markdown("**:blue[6-2-1. 훈련데이터 Mode 값]**")
                                st.dataframe(df_mode.reset_index().rename(columns = {'index':'Columns', 0:'Columns Mode Values'}))

                                train[object_columns] = train[object_columns].fillna(df_mode)
                                test[object_columns] = test[object_columns].fillna(df_mode)

                        else:
                            groupby_columns3 = st.multiselect("Groupby 컬럼을 선택하세요", train.columns.to_list(), default=train.columns.to_list()[0], help="단일값만 있을 경우 오류가 발생합니다. 그룹화가 가능하게 값을 설정하세요.", key='groupby_columns3')
                            train_object_columns = train.columns.to_list()
                            train_object_columns = [i for i in train_object_columns if i not in groupby_columns3]
                           


                            train_mode = train.groupby(groupby_columns3)[train_object_columns].agg(lambda x: x.mode().iloc[0]).reset_index()
                            st.write('\n')
                            st.markdown("**:blue[6-2-2. 훈련데이터 문자형 Mode 값]**")
                            st.write(train_mode)


                            train_update = train.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            train_update = train_update.set_index(train.index)
                            train = train_update


                            for i in train_object_columns: 
                                a = i + '_mode'
                                train[i] = train[i].fillna(train[a])
                                train = train.drop(columns=[a])
      
                            test_update = test.merge(train_mode, on=groupby_columns3, how='left', suffixes=('', '_mode'))
                            test_update = test_update.set_index(test.index)
                            test = test_update

                            for i in train_object_columns: 
                                a = i + '_mode'
                                test[i] = test[i].fillna(test[a])
                                test = test.drop(columns=[a])

                           
                    
                           
                        st.write('\n')
                        st.markdown("**:blue[6-2-2. 최종변경된 데이터 값]**")
                        st.write('\n')
                        st.markdown("**:black[① Train]**")
                        st.dataframe(train)
                        st.write('\n')

                        st.markdown("**:black[② Test]**")
                        st.dataframe(test)


            
                else:
                    pass
    return train, test, val


# 6. numeric encoder 

def numeric_columns_encoding(train, test, val):
    train_df_numeric = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
    if val is not None:
        st.write('\n')
        select_numeric_encoding = st.selectbox("인코딩을 사용하시겠습니까?",('No', 'Yes'))                     
        if select_numeric_encoding == 'Yes':
            st.write('\n')
            st.markdown("**:blue[7-1. Scaling 방법선택]**")
            scaler_method = st.radio("Scaling 종류",('Standard', 'Normalize', 'MinMax','MaxAbs','Robust'), horizontal=True, help= 'std: (평균 = 0, 분산 = 1), Normalize: (유클리드 거리=1), MinMax:(0 에서 1 사이), MaxAbs:(-1 에서 1 사이), Robust: (중앙값=0, IQE=1)')
            scaler_select_columns = st.multiselect('컬럼을 선택하세요.', train_df_numeric, train_df_numeric, key='<scaler_select_columns>')
            if len(scaler_select_columns) >0:

                if scaler_method =='Standard':

                    standard_scaler = StandardScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index
                    val_index = val.index

                    train[scaler_select_columns] = standard_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = standard_scaler.transform(test[scaler_select_columns])
                    val[scaler_select_columns] = standard_scaler.transform(val[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    val.index = val_index
            
                elif scaler_method =='Normalize':
                    
                    Normalize_scaler = Normalizer()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index
                    val_index = val.index

                    train[scaler_select_columns] = Normalize_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Normalize_scaler.transform(test[scaler_select_columns])
                    val[scaler_select_columns] = Normalize_scaler.transform(val[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    val.index = val_index

                
                elif scaler_method =='MinMax':
                    
                    Minmax_scaler = MinMaxScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index
                    val_index = val.index

                    train[scaler_select_columns] = Minmax_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Minmax_scaler.transform(test[scaler_select_columns])
                    val[scaler_select_columns] = Minmax_scaler.transform(val[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    val.index = val_index
                
                elif scaler_method =='MaxAbs':
                    
                    Maxabs_scaler = MaxAbsScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index
                    val_index = val.index

                    train[scaler_select_columns] = Maxabs_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Maxabs_scaler.transform(test[scaler_select_columns])
                    val[scaler_select_columns] = Maxabs_scaler.transform(val[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    val.index = val_index

                elif scaler_method =='Robust':
                    
                    Robust_scaler = RobustScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index
                    val_index = val.index

                    train[scaler_select_columns] = Robust_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Robust_scaler.transform(test[scaler_select_columns])
                    val[scaler_select_columns] = Robust_scaler.transform(val[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    val.index = val_index
                    
            else:
                st.warning("적용할 컬럼을 선택하세요.")

                

            st.write('\n')
            st.markdown("**:blue[7-2. 최종변경된 데이터 값]**")
            st.write('\n')
            st.markdown("**:black[① Train]**")
            st.dataframe(train)
            st.write('\n')

            st.markdown("**:black[② Test]**")
            st.dataframe(test)    

            st.write('\n')
            st.markdown("**:black[③ Val]**")
            st.dataframe(val)    


        else:
            st.info("변경된 값이 없습니다.")   
        
        return train, test, val

    else: # val is none
        st.write('\n')
        select_numeric_encoding = st.selectbox("인코딩을 사용하시겠습니까?",('Yes', 'No'))                     
        if select_numeric_encoding == 'Yes':
            st.write('\n')
            st.markdown("**:blue[7-1. Scaling 방법선택]**")
            scaler_method = st.radio("Scaling 종류",('Standard', 'Normalize', 'MinMax','MaxAbs','Robust'), horizontal=True, help= 'std: (평균 = 0, 분산 = 1), Normalize: (유클리드 거리=1), MinMax:(0 에서 1 사이), MaxAbs:(-1 에서 1 사이), Robust: (중앙값=0, IQE=1)')
            scaler_select_columns = st.multiselect('컬럼을 선택하세요.', train_df_numeric, train_df_numeric, key='<scaler_select_columns>')
            if len(scaler_select_columns) >0:

                if scaler_method =='Standard':

                    standard_scaler = StandardScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index

                    train[scaler_select_columns] = standard_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = standard_scaler.transform(test[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
            
                elif scaler_method =='Normalize':
                    
                    Normalize_scaler = Normalizer()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index

                    train[scaler_select_columns] = Normalize_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Normalize_scaler.transform(test[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index

                
                elif scaler_method =='MinMax':
                    
                    Minmax_scaler = MinMaxScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index

                    train[scaler_select_columns] = Minmax_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Minmax_scaler.transform(test[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                
                elif scaler_method =='MaxAbs':
                    
                    Maxabs_scaler = MaxAbsScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index

                    train[scaler_select_columns] = Maxabs_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Maxabs_scaler.transform(test[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index

                elif scaler_method =='Robust':
                    
                    Robust_scaler = RobustScaler()

                    # 스케일 변환 이전에 인덱스 저장
                    train_index = train.index
                    test_index = test.index

                    train[scaler_select_columns] = Robust_scaler.fit_transform(train[scaler_select_columns])            
                    test[scaler_select_columns] = Robust_scaler.transform(test[scaler_select_columns])


                    # 인덱스 재설정
                    train.index = train_index
                    test.index = test_index
                    
            else:
                st.warning("적용할 컬럼을 선택하세요.")

                

            st.write('\n')
            st.markdown("**:blue[7-2. 최종변경된 데이터 값]**")
            st.write('\n')
            st.markdown("**:black[① Train]**")
            st.dataframe(train)
            st.write('\n')

            st.markdown("**:black[② Test]**")
            st.dataframe(test)    



        else:
            st.info("변경된 값이 없습니다.")
        return train, test   

def Polynomial_encoding(train, test, val):
    train_df_numeric = train.select_dtypes(exclude = ['object', 'datetime']).columns.to_list()
    categorical_columns = train.select_dtypes(include=['object']).columns.to_list()
    # 나중에 interaction_only, include_bias 를 작성할 수 있는 것 추가
    if val is not None:
        st.write('\n')
        select_polynomial = st.selectbox("다항회귀를 사용하시겠습니까?",('No', 'Yes'), help='수치형 컬럼이 있을경우에만 사용가능합니다.')
        if select_polynomial == 'Yes':      
            poly_select_columns = st.multiselect('적용할 컬럼을 선택하세요.', train_df_numeric)

            if len(poly_select_columns) > 0:
                number = st.slider("차수를 선택하세요.", min_value=2, max_value=10, format='%d')
                
                train_index = train.index
                test_index = test.index
                val_index = val.index

                poly_scaler = PolynomialFeatures(degree=number)
                poly_features_train = poly_scaler.fit_transform(train[poly_select_columns])
                poly_features_test = poly_scaler.transform(test[poly_select_columns])
                poly_features_val = poly_scaler.transform(val[poly_select_columns])

                poly_columns = poly_scaler.get_feature_names_out(poly_select_columns)

                poly_data_train = pd.DataFrame(poly_features_train, columns=poly_columns, index=train_index)
                poly_data_test = pd.DataFrame(poly_features_test, columns=poly_columns, index=test_index)
                poly_data_val = pd.DataFrame(poly_features_val, columns=poly_columns, index=val_index)

                ## 다항식 피처를 기존 데이터프레임과 병합
                train = pd.concat([train.drop(poly_select_columns, axis=1), poly_data_train], axis=1)
                test = pd.concat([test.drop(poly_select_columns, axis=1), poly_data_test], axis=1)
                val = pd.concat([val.drop(poly_select_columns, axis=1), poly_data_val], axis=1)

                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.dataframe(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.dataframe(test)    
                st.write('\n')
                st.markdown("**:black[② Val]**")
                st.dataframe(val)  

            else: 
                st.warning("적용할 컬럼을 선택해주세요.")

     



        else: # poly = no
            st.info("변경된 값이 없습니다.")   
        


            
    
    else:
        st.write('\n')
        select_polynomial = st.selectbox("다항회귀를 사용하시겠습니까?",('No', 'Yes'), help='수치형 컬럼이 있을경우에만 사용가능합니다.')

        if select_polynomial == 'Yes':      
            poly_select_columns = st.multiselect('적용할 컬럼을 선택하세요.', train_df_numeric)

            if len(poly_select_columns) > 0:
                number = st.slider("차수를 선택하세요.", min_value=2, max_value=10, format='%d')
                
                train_index = train.index
                test_index = test.index

                poly_scaler = PolynomialFeatures(degree=number)
                poly_features_train = poly_scaler.fit_transform(train[poly_select_columns])
                poly_features_test = poly_scaler.transform(test[poly_select_columns])

                poly_columns = poly_scaler.get_feature_names_out(poly_select_columns)

                poly_data_train = pd.DataFrame(poly_features_train, columns=poly_columns, index=train_index)
                poly_data_test = pd.DataFrame(poly_features_test, columns=poly_columns, index=test_index)

                ## 다항식 피처를 기존 데이터프레임과 병합 여기서부터 그룹 맥스
                train = pd.concat([train.drop(poly_select_columns, axis=1), poly_data_train], axis=1)
                test = pd.concat([test.drop(poly_select_columns, axis=1), poly_data_test], axis=1)

                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.dataframe(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.dataframe(test)    
                st.write('\n')

            else: 
                st.warning("적용할 컬럼을 선택해주세요.")

            




        else: # poly = no
            st.info("변경된 값이 없습니다.")   
        

    return train, test, val


# 9. Label Incoder | Onehotencoder 
# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
def label_onehot_encoder(train, test, val):
    if val is not None:
        st.write('\n')
        select_encoding = st.selectbox("인코딩을 사용하시겠습니까?",('No', 'Yes'), key='select_encoding1')
        if select_encoding == 'Yes':
            st.write('\n')
            encoding_method = st.radio('방법을 선택하세요.', ('Ordinal Encoder', 'Onehot Encoder'), horizontal=True)
            if encoding_method == "Ordinal Encoder":
                object_columns = train.select_dtypes(include=object).columns.to_list()
                select_columns_encoding = st.multiselect('적용할 컬럼을 선택하세요.', object_columns, default=object_columns[0])
                                
                if len(select_columns_encoding) > 0:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    train[select_columns_encoding] = oe.fit_transform(train[select_columns_encoding])
                    test[select_columns_encoding] = oe.transform(test[select_columns_encoding])
                    val[select_columns_encoding] = oe.transform(val[select_columns_encoding])
                    
                else:
                    st.warning("적용할 컬럼을 선택하세요.")
                
                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[③ Test]**")
                st.write(test)
                st.write('\n')
                st.markdown("**:black[③ Val]**")
                st.write(val)

                st.write('\n')
                st.markdown("**:blue[8-1. 최종 인코딩 된 데이터 값]**")
                categories = oe.categories_
                st.write(categories)




            else:
                object_columns = train.select_dtypes(include=object).columns.to_list()
                select_columns_encoding = st.multiselect('적용할 컬럼을 선택하세요.', object_columns, default=object_columns[0])
                                
                if len(select_columns_encoding) > 0:
                    ohe = OneHotEncoder(handle_unknown='ignore')
                    
                    train_encoded = ohe.fit_transform(train[select_columns_encoding].astype(str)).toarray()
                    test_encoded = ohe.transform(test[select_columns_encoding].astype(str)).toarray()
                    val_encoded = ohe.transform(val[select_columns_encoding].astype(str)).toarray()
                    
                    train_encoded_df = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out(select_columns_encoding))
                    test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(select_columns_encoding))
                    val_encoded_df = pd.DataFrame(val_encoded, columns=ohe.get_feature_names_out(select_columns_encoding))

                    # 기존의 선택한 컬럼들을 삭제하고 변환된 데이터를 할당
                    train.drop(columns=select_columns_encoding, inplace=True)
                    test.drop(columns=select_columns_encoding, inplace=True)
                    val.drop(columns=select_columns_encoding, inplace=True)

                    train_encoded_df.index = train.index
                    test_encoded_df.index = test.index
                    val_encoded_df.index = val.index

                    train = train.join(train_encoded_df)
                    test = test.join(test_encoded_df)
                    val = val.join(val_encoded_df)
                    
                else:
                    st.warning("적용할 컬럼을 선택하세요.")
                
                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')
                st.markdown("**:black[③ Val]**")
                st.write(val)

               

        else:
            st.info("변경된 값이 없습니다.")   
   
    else:
        st.write('\n')
        select_encoding = st.selectbox("인코딩을 사용하시겠습니까?",('No', 'Yes'))
        if select_encoding == 'Yes':
            st.write('\n')
            encoding_method = st.radio('방법을 선택하세요.', ('Ordinal Encoder', 'Onehot Encoder'), horizontal=True)
            if encoding_method == "Ordinal Encoder":
                object_columns = train.select_dtypes(include=object).columns.to_list()
                select_columns_encoding = st.multiselect('적용할 컬럼을 선택하세요.', object_columns, default=object_columns[0])
                                
                if len(select_columns_encoding) > 0:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    train[select_columns_encoding] = oe.fit_transform(train[select_columns_encoding])
                    test[select_columns_encoding] = oe.transform(test[select_columns_encoding])
                    
                else:
                    st.warning("적용할 컬럼을 선택하세요.")
                
                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')                                
                st.markdown("**:blue[8-1. 최종 인코딩 된 데이터 값]**")
                categories = oe.categories_
                st.write(categories)




            else:
                object_columns = train.select_dtypes(include=object).columns.to_list()
                select_columns_encoding = st.multiselect('적용할 컬럼을 선택하세요.', object_columns, default=object_columns[0])
                                
                if len(select_columns_encoding) > 0:
                    ohe = OneHotEncoder(handle_unknown='ignore')
                    
                    train_encoded = ohe.fit_transform(train[select_columns_encoding].astype(str)).toarray()
                    test_encoded = ohe.transform(test[select_columns_encoding].astype(str)).toarray()
                    
                    train_encoded_df = pd.DataFrame(train_encoded, columns=ohe.get_feature_names_out(select_columns_encoding))
                    test_encoded_df = pd.DataFrame(test_encoded, columns=ohe.get_feature_names_out(select_columns_encoding))

                    # 기존의 선택한 컬럼들을 삭제하고 변환된 데이터를 할당
                    train.drop(columns=select_columns_encoding, inplace=True)
                    test.drop(columns=select_columns_encoding, inplace=True)

                    train_encoded_df.index = train.index
                    test_encoded_df.index = test.index

                    train = train.join(train_encoded_df)
                    test = test.join(test_encoded_df)
                    
                else:
                    st.warning("적용할 컬럼을 선택하세요.")
                
                st.write('\n')
                st.markdown("**:blue[8-1. 최종변경된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')

               

        else:
            st.info("변경된 값이 없습니다.")   
    return train, test, val


# drop_outlier 이상치 제거 부분 생각이 필요 값을 채우기 전에 outlier를 계산 후 작성 필요.

#def drop_outlier(train, test, val):
#    numeric_train = train.select_dtypes(exclude=['object', 'datetime']).columns.to_list()
#
#    if val is not None:
#        st.write('\n')
#        select_ouliter = st.selectbox("이상치 제거를 하시겠습니까?",('No', 'Yes'), help='이상치 제거는 훈련데이터에서만 적용됩니다.')
#        if select_ouliter == 'Yes':
#            st.write('\n')
#            outlier_method = st.radio("이상치 제거 방법을 선택하세요.",('Z-score', 'IQR'), horizontal=True)
#            st.write('\n')
#            oulier_columns = st.multiselect("이상치를 제거할 컬럼을 선택하세요", numeric_train)
#            
#            if len(oulier_columns) > 0:
#
#                if outlier_method == 'Z-score':
#                    threshold = st.number_input("Z-score 임계값", value=3.0)
#                    z_scores = np.abs((train[oulier_columns] - train[oulier_columns].mean()) / train[oulier_columns].std())
#                    filtered_data = train[(np.abs(z_scores) <= threshold).all(axis=1)]
#                    train = filtered_data
#                    st.write(z_scores)
#                    st.write(train)
#                    st.write(train.isnull().sum())
#                    return train, test, val
#            
#            else:
#                st.info("이상치 제거가 필요한 컬럼을 선택하세요.")
#
#
#        else:
#            st.info("변경된 값이 없습니다.")
#            st.write(train)
#    # val in none
#    else:
#        pass
#    return train, test, val
# 프로그램 시작 부

def generate_download_button(train, test, val):
    if val is not None:
        st.write('\n')
        train = train.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        test = test.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        val = val.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        st.write('\n')
        st.markdown("**:blue[버튼을 눌러 최종 데이터를 다운로드 하세요.]**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button("Train", train, file_name='train.csv')
        with col2:
            st.download_button("Test", test, file_name='test.csv')
        with col3:
            st.download_button("Val", val, file_name='val.csv')
    else:
        st.write('\n')
        train = train.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        test = test.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
        st.write('\n')
        st.markdown("**:blue[버튼을 눌러 최종 데이터를 다운로드 하세요.]**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button("Train", train, file_name='train.csv')
        with col2:
            st.download_button("Test", test, file_name='test.csv')
        with col3:
            st.write('\n')
    return train, test, val

# 화면노출
def preprocessing_app():
    st.title("🚧 Pre-processing")
    st.write('\n')
    st.subheader("1. 파일 업로드")
    upload_file = st.file_uploader("", type=['xlsx', 'xls', 'csv'], accept_multiple_files=False)
    
    if upload_file is not None:
        try:
            # 데이터 불러오기
            df, columns = load_dataframe(upload_file=upload_file)
            st.markdown("---")
            st.write('\n')
            st.subheader("2. 전체 데이터 탐색")
            st.write('\n')
            st.write('\n')
            try:
                st.dataframe(df) # 함수
            except Exception as e:
                print(e)
    
    
            st.markdown("---")
            st.subheader("3. 필요없는 컬럼 삭제")
            try:
                features = drop_df(df)
            except Exception as e:
                print(e)
                
            st.markdown("---")
            st.subheader("4. 빈값을 가진 컬럼 삭제")
            try:
                features = Drop_na(features)
            except Exception as e:
                print(e)    
    
            st.markdown("---")
            st.subheader("5. 특수문자 제거")
            try:
                features = special_str_drop(features)
            except Exception as e:
                print(e)    
            #
            st.markdown("---")
            st.subheader("6. 훈련 & 테스트 & 검증 데이터 분리")
            try:
                train, test, val = split_train_test_split(features)
            except Exception as e:
                print(e)      
    #
            st.markdown("---")
            st.subheader("7. 빈값 채우기")
            try:
                train,test,val = fill_na(train, test, val)
                
            except Exception as e:
                print(e)  
            #
            #
            #st.markdown("---")
            #st.subheader("7. 이상치 값 제거")
            #try:
            #    train, test, val = drop_outlier(train, test, val)
            #except Exception as e:
            #    print(e)
    #
            st.markdown("---")
            st.subheader("8. 수치형 타입 인코딩")
            try:
                train, test, val = numeric_columns_encoding(train, test, val)
               
            except Exception as e:
                print(e) 
            
            st.markdown("---")
            st.subheader("9. 다항회귀(Polynomial Features) 생성")
            try:
                train, test, val = Polynomial_encoding(train, test, val)
               
            except Exception as e:
                print(e) 
            #    
            st.markdown("---")
            st.subheader("10. 문자형 타입 인코딩")
            try:
                train, test, val = label_onehot_encoder(train, test, val)
            except Exception as e:
                print(e) 
    
            st.markdown("---")
            st.subheader("11. 최종완료 파일 다운로드")
            try:
                train, test, val = generate_download_button(train, test, val)
            except Exception as e:
                print(e) 
    #
    #
          #
        except Exception as e:
                print(e)
    #
    #
    else:
        try:
            sample_data = st.checkbox('샘플데이터 사용', value=True)
            if sample_data:
                st.error('SAMPLE 데이터로 확인중', icon="⚠️")
                # 데이터 불러오기
                df, columns = load_dataframe('sample_data/sample_df.csv')
                st.markdown("---")
                st.write('\n')
                st.subheader("2. 전체 데이터 탐색")
                st.write('\n')
                st.write('\n')
                try:
                    st.dataframe(df) # 함수
                except Exception as e:
                    print(e)


                st.markdown("---")
                st.subheader("3. 필요없는 컬럼 삭제")
                try:
                    features = drop_df(df)
                except Exception as e:
                    print(e)

                st.markdown("---")
                st.subheader("4. 빈값을 가진 컬럼 삭제")
                try:
                    features = Drop_na(features)
                except Exception as e:
                    print(e)    

                st.markdown("---")
                st.subheader("5. 특수문자 제거")
                try:
                    features = special_str_drop(features)
                except Exception as e:
                    print(e)    
                #
                st.markdown("---")
                st.subheader("6. 훈련 & 테스트 & 검증 데이터 분리")
                try:
                    train, test, val = split_train_test_split(features)
                except Exception as e:
                    print(e)      
    #   
                st.markdown("---")
                st.subheader("7. 빈값 채우기")
                try:
                    train,test,val = fill_na(train, test, val)

                except Exception as e:
                    print(e)  
                #
                #
                #st.markdown("---")
                #st.subheader("7. 이상치 값 제거")
                #try:
                #    train, test, val = drop_outlier(train, test, val)
                #except Exception as e:
                #    print(e)
    #   
                st.markdown("---")
                st.subheader("8. 수치형 타입 인코딩")
                try:
                    train, test, val = numeric_columns_encoding(train, test, val)

                except Exception as e:
                    print(e) 

                st.markdown("---")
                st.subheader("9. 다항회귀(Polynomial Features) 생성")
                try:
                    train, test, val = Polynomial_encoding(train, test, val)

                except Exception as e:
                    print(e) 
                #    
                st.markdown("---")
                st.subheader("10. 문자형 타입 인코딩")
                try:
                    train, test, val = label_onehot_encoder(train, test, val)
                except Exception as e:
                    print(e) 

                st.markdown("---")
                st.subheader("11. 최종완료 파일 다운로드")
                try:
                    train, test, val = generate_download_button(train, test, val)
                except Exception as e:
                    print(e) 
            else:
                st.error("데이터를 넣어주세요.")
    #
          #
        except Exception as e:
                print(e)



