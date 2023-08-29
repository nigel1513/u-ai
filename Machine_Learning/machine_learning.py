import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import re
import joblib
import scipy.stats as stats
from sklearn.inspection import plot_partial_dependence

import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier


import catboost as cb

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import os

# 프로그램 함수부
# 1. 업로드 함수
def file_upload():
    st.write('\n')
    st.write('\n') 
    sample_data = st.checkbox('샘플데이터 사용', value=True)
    if sample_data:     
        check_validation = st.selectbox(
        '검증용 데이터를 가지고 있으십니까?',
        ('No', 'Yes'), help="YES일 경우 Train, Test, Val")
        st.write('\n')  
        st.write('\n')  

        if check_validation == 'Yes':
            train_file = 'a'
            test_file = 'b'
            val_file = 'c'
                    
            if train_file and test_file and val_file is not None:

                train = pd.read_csv('sample_data/a.csv', encoding='utf-8')
                test = pd.read_csv('sample_data/a.csv', encoding='utf-8')
                val = pd.read_csv('sample_data/a.csv', encoding='utf-8')

                train_columns = train.columns.to_list()
                train_columns.insert(0, 'None')
                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-2. Index 컬럼 선택]**")

                index_select = st.selectbox('컬럼을 선택하세요.', train_columns, help="기존 Index를 사용할 경우 None 선택")
                if index_select == 'None':
                    st.success("Index 변경사항이 없습니다.")

                else:
                    train = train.set_index(index_select, drop=True)              
                    test = test.set_index(index_select, drop=True)
                    val = val.set_index(index_select, drop=True)         
                    st.success("Index 컬럼:" + "  " + "[" + str(index_select) + ']' )

                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-3. 업로드 된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')
                st.markdown("**:black[③ Val]**")
                st.write(val)
                st.write('\n')
                columns_train = set(train.columns)
                columns_test = set(test.columns)
                columns_val = set(val.columns)
                if columns_train == columns_test == columns_val:
                    st.success("Train & Test & Val 컬럼이 모두 일치합니다.")
                else:
                    st.error("Train & Test & Val 컬럼이 일치하지 않습니다.")
            else:
                st.error("Train & Test & Val 데이터를 업로드해주세요.")

        elif check_validation == 'No':


            train_file = 'a'
            test_file = 'b'

            if train_file and test_file is not None:

                train = pd.read_csv('sample_data/a.csv', encoding='utf-8')
                test = pd.read_csv('sample_data/a.csv', encoding='utf-8')
                val = None
                columns_val = None

                train_columns = train.columns.to_list()
                train_columns.insert(0, 'None')
                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-2. Index 컬럼 선택]**")

                index_select = st.selectbox('컬럼을 선택하세요.', train_columns, help="기존 Index를 사용할 경우 None 선택")
                if index_select == 'None':
                    st.success("Index 변경사항이 없습니다.")

                else:
                    train = train.set_index(index_select, drop=True)              
                    test = test.set_index(index_select, drop=True)
                    st.success("Index 컬럼:" + "  " + "[" + str(index_select) + ']' )

                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-3. 업로드 된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')

                columns_train = set(train.columns)
                columns_test = set(test.columns)

                if columns_train == columns_test:
                    st.success("Train & Test 컬럼이 모두 일치합니다.")
                else:
                    st.error("Train & Test 컬럼이 일치하지 않습니다.")

            else:
                st.error("Train & Test 데이터를 업로드해주세요.")
    else:
        check_validation = st.selectbox(
        '검증용 데이터를 가지고 있으십니까?',
        ('No', 'Yes'), help="YES일 경우 Train, Test, Val")
        st.write('\n')  
        st.write('\n')  
        st.markdown("**:blue[1-1. 데이터를 업로드해 주세요.]**")
        st.write('\n')  

        if check_validation == 'Yes':
            col1, col2, col3 = st.columns(3)

            with col1:
                train_file = st.file_uploader("훈련데이터 업로드", type=["csv"])
            with col2:
                test_file = st.file_uploader("테스트데이터 업로드", type=["csv"])
            with col3:
                val_file = st.file_uploader("검증데이터 업로드", type=["csv"])

            if train_file and test_file and val_file is not None:

                train = pd.read_csv(train_file, encoding='utf-8')
                test = pd.read_csv(test_file, encoding='utf-8')
                val = pd.read_csv(val_file, encoding='utf-8')

                train_columns = train.columns.to_list()
                train_columns.insert(0, 'None')
                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-2. Index 컬럼 선택]**")

                index_select = st.selectbox('컬럼을 선택하세요.', train_columns, help="기존 Index를 사용할 경우 None 선택")
                if index_select == 'None':
                    st.success("Index 변경사항이 없습니다.")

                else:
                    train = train.set_index(index_select, drop=True)              
                    test = test.set_index(index_select, drop=True)
                    val = val.set_index(index_select, drop=True)         
                    st.success("Index 컬럼:" + "  " + "[" + str(index_select) + ']' )

                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-3. 업로드 된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')
                st.markdown("**:black[③ Val]**")
                st.write(val)
                st.write('\n')
                columns_train = set(train.columns)
                columns_test = set(test.columns)
                columns_val = set(val.columns)
                if columns_train == columns_test == columns_val:
                    st.success("Train & Test & Val 컬럼이 모두 일치합니다.")
                else:
                    st.error("Train & Test & Val 컬럼이 일치하지 않습니다.")
            else:
                st.error("Train & Test & Val 데이터를 업로드해주세요.")

        elif check_validation == 'No':

            col1, col2 = st.columns(2)

            with col1:
                train_file = st.file_uploader("훈련데이터 업로드", type=["csv"])
            with col2:
                test_file = st.file_uploader("테스트데이터 업로드", type=["csv"])

            if train_file and test_file is not None:

                train = pd.read_csv(train_file, encoding='utf-8')
                test = pd.read_csv(test_file, encoding='utf-8')
                val = None
                columns_val = None

                train_columns = train.columns.to_list()
                train_columns.insert(0, 'None')
                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-2. Index 컬럼 선택]**")

                index_select = st.selectbox('컬럼을 선택하세요.', train_columns, help="기존 Index를 사용할 경우 None 선택")
                if index_select == 'None':
                    st.success("Index 변경사항이 없습니다.")

                else:
                    train = train.set_index(index_select, drop=True)              
                    test = test.set_index(index_select, drop=True)
                    st.success("Index 컬럼:" + "  " + "[" + str(index_select) + ']' )

                st.write('\n')
                st.write('\n')
                st.markdown("**:blue[1-3. 업로드 된 데이터 값]**")
                st.write('\n')
                st.markdown("**:black[① Train]**")
                st.write(train)
                st.write('\n')
                st.markdown("**:black[② Test]**")
                st.write(test)
                st.write('\n')

                columns_train = set(train.columns)
                columns_test = set(test.columns)

                if columns_train == columns_test:
                    st.success("Train & Test 컬럼이 모두 일치합니다.")
                else:
                    st.error("Train & Test 컬럼이 일치하지 않습니다.")

            else:
                st.error("Train & Test 데이터를 업로드해주세요.")



    return sample_data, train, test, val, columns_train, columns_test, columns_val

# 2. 타겟값 분리 함수 적용
def drop_target(train, test, val, columns_train, columns_test, columns_val):
    select_target = st.selectbox(
    '타겟 컬럼을 선택해주세요.',
    train.columns.to_list(), help='타겟컬럼이란 예측하고 싶은 컬럼을 의미합니다.')

    train = train
    test = test
    val = val

    if val is not None:
        if columns_train == columns_test == columns_val:
            train_target = train[select_target]
            train = train.drop(columns=[select_target])
            test_target = test[select_target]
            test = test.drop(columns=[select_target])
            val_target = val[select_target]
            val = val.drop(columns=[select_target])
            
            st.write('\n')
            st.markdown("**:blue[2-1. 전체 타겟값 확인.]**")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**:black[① Train Target]**")
                st.write(train_target)

            with col2:
                st.markdown("**:black[② Test Target]**")
                st.write(test_target)
                
            with col3:
                st.markdown("**:black[② Val Target]**")
                st.write(val_target)
        else:
            st.error("Train & Test & Val 데이터 컬럼이 일치하지 않습니다.")
    else:
        if columns_train == columns_test:
            train_target = train[select_target]
            train = train.drop(columns=[select_target])
            test_target = test[select_target]
            test = test.drop(columns=[select_target])
            val_target = None
            val = None
            
            st.write('\n')
            st.markdown("**:blue[2-1. 전체 타겟값 확인.]**")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**:black[① Train Target]**")
                st.write(train_target)

            with col2:
                st.markdown("**:black[② Test Target]**")
                st.write(test_target)

        else:
            st.error("Train & Test & Val 데이터 컬럼이 일치하지 않습니다.")
       
        
    
    return train, test, val, train_target, test_target, val_target





def algorithm_select(sample_data, train, test, val, train_target, test_target, val_target):

    if sample_data:

        st.write('\n')
        task = st.radio("**:blue[3-1. 예측 태스크를 선택해주세요.]**", ("분류", "회귀"), horizontal=True)
        models = ["XGBoost", "LightGBM", "CatBoost"]
        model_select = st.selectbox('사용할 모델을 선택하세요', models)
        st.info("Hyperparameters는 RandomizedSearchCV를 활용합니다.")

        if val is not None:

            train = train
            test = test
            val = val
            train_target = train_target
            test_target = test_target
            val_target = val_target

            if task == '분류':
                if model_select == 'XGBoost':
                    model = XGBClassifier(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()



                elif model_select == 'LightGBM':

                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val.columns = val.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)
                    val_target = pd.DataFrame(val_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val_target.columns = val_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()
                    val_target = val_target.squeeze()

                    model = LGBMClassifier(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


                elif  model_select == 'CatBoost':


                    model = CatBoostClassifier(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostClassifier(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


            elif task == '회귀':
                if model_select == 'XGBoost':
                    model = XGBRegressor(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()


                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val.columns = val.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)
                    val_target = pd.DataFrame(val_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val_target.columns = val_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()
                    val_target = val_target.squeeze()

                    model = LGBMRegressor(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostRegressor(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostRegressor(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']

                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


        elif val is None:
            train = train
            test = test
            val = None
            train_target = train_target
            test_target = test_target
            val_target = None
            val_pred = None

            if task == '분류':
                if model_select == 'XGBoost':
                    model = XGBClassifier(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()



                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()

                    model = LGBMClassifier(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test)

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostClassifier(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostClassifier(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


            elif task == '회귀':
                if model_select == 'XGBoost':
                    model = XGBRegressor(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()

                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()

                    model = LGBMRegressor(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostRegressor(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=5, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostRegressor(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']

                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

    # sample이 아닐 경우 
    else:
        st.write('\n')
        task = st.radio("**:blue[3-1. 예측 태스크를 선택해주세요.]**", ("분류", "회귀"), horizontal=True)
        models = ["XGBoost", "LightGBM", "CatBoost"]
        model_select = st.selectbox('사용할 모델을 선택하세요', models)
        st.info("Hyperparameters는 RandomizedSearchCV를 활용합니다.")

        if val is not None:

            train = train
            test = test
            val = val
            train_target = train_target
            test_target = test_target
            val_target = val_target

            if task == '분류':
                if model_select == 'XGBoost':
                    model = XGBClassifier(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()



                elif model_select == 'LightGBM':

                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val.columns = val.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)
                    val_target = pd.DataFrame(val_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val_target.columns = val_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()
                    val_target = val_target.squeeze()

                    model = LGBMClassifier(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


                elif  model_select == 'CatBoost':


                    model = CatBoostClassifier(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostClassifier(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


            elif task == '회귀':
                if model_select == 'XGBoost':
                    model = XGBRegressor(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()


                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val.columns = val.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)
                    val_target = pd.DataFrame(val_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    val_target.columns = val_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()
                    val_target = val_target.squeeze()

                    model = LGBMRegressor(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostRegressor(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostRegressor(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = best_model.predict(val) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']

                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


        elif val is None:
            train = train
            test = test
            val = None
            train_target = train_target
            test_target = test_target
            val_target = None
            val_pred = None

            if task == '분류':
                if model_select == 'XGBoost':
                    model = XGBClassifier(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()



                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()

                    model = LGBMClassifier(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMClassifier(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test)

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostClassifier(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring='accuracy', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostClassifier(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_score'] = results['mean_test_score']
                    parameter_bests['std_test_score'] = results['std_test_score']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_score', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_


            elif task == '회귀':
                if model_select == 'XGBoost':
                    model = XGBRegressor(random_state=2023)
                    param_dist = {
                    'learning_rate': np.arange(0.01,1,0.01),
                    'max_depth': np.arange(10, 100, 10),
                    'min_child_weight': np.arange(1,6,1),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1),
                    'gamma': [0, 100, 10],
                    'n_estimators':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = XGBRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())

                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.get_booster()

                elif model_select == 'LightGBM':
                    train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = pd.DataFrame(train_target)
                    test_target = pd.DataFrame(test_target)

                    train_target.columns = train_target.columns.str.replace('[^A-Za-z0-9_]+', '')
                    test_target.columns = test_target.columns.str.replace('[^A-Za-z0-9_]+', '')

                    train_target = train_target.squeeze()
                    test_target = test_target.squeeze()

                    model = LGBMRegressor(random_state=2023)
                    param_dist = {
                    'num_leaves': np.arange(0, 31, 1),
                    'learning_rate': np.arange(0.01,1,0.01),
                    'n_estimators':np.arange(10, 1000, step=50),
                    'max_depth': np.arange(10, 100, 10),
                    'subsample': np.arange(0.1,1,0.1),
                    'colsample_bytree': np.arange(0.1,1,0.1)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = LGBMRegressor(**random_search.best_params_, random_state=2023)
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']


                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_

                elif  model_select == 'CatBoost':
                    model = CatBoostRegressor(random_state=2023, bootstrap_type = 'Bernoulli')
                    param_dist = {
                    'learning_rate':np.arange(0.001, 0.1, 0.002),
                    'depth': np.arange(1, 16, 1),
                    'l2_leaf_reg': np.arange(1,10,1), 
                    'subsample': np.arange(0.01, 1.0, 0.01),
                    'colsample_bylevel': np.arange(0.01, 1.0, 0.01),
                    'iterations':np.arange(10, 1000, step=50)}

                    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                        n_iter=30, cv=5, verbose=1, n_jobs=-1, scoring={'mse':'neg_mean_squared_error', 'r2':'r2'}, refit='mse', random_state=2023)

                    random_search.fit(train, train_target)

                    best_model = CatBoostRegressor(**random_search.best_params_, random_state=2023, bootstrap_type = 'Bernoulli')
                    best_model.fit(train, train_target)

                    train_pred = best_model.predict(train)  
                    test_pred = best_model.predict(test) 
                    val_pred = None

                    results = random_search.cv_results_
                    param_keys = list(param_dist.keys())
                    parameter_bests = pd.DataFrame({param: results['param_' + param] for param in param_keys})
                    parameter_bests = parameter_bests.apply(pd.to_numeric, errors='coerce')
                    parameter_bests['mean_test_mse'] = results['mean_test_mse'] * -1
                    parameter_bests['mean_test_r2'] = results['mean_test_r2']
                    parameter_bests['std_test_mse'] = results['std_test_mse'] * -1
                    parameter_bests['std_test_r2'] = results['std_test_r2']

                    st.write('\n')
                    st.write('\n')
                    st.markdown("**:blue[3-2. 하이퍼파라미터별 스코어]**")
                    parameter_best_sorted = parameter_bests.sort_values(by='mean_test_mse', ascending=False)
                    st.write(parameter_best_sorted)

                    booster = best_model.feature_importances_





    return task, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred, model_select


def xgboost_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    if val_pred is not None:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_dict = booster.get_score(importance_type='total_gain')
            feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()

            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)
            
            st.write('\n')

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)
    


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_dict = booster.get_score(importance_type='total_gain')
            feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)      

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")
            
    else:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_dict = booster.get_score(importance_type='total_gain')
            feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')                


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_dict = booster.get_score(importance_type='total_gain')
            feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")

def xgboost_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    st.write('\n')
    st.write('\n')

    st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
    feature_importance_dict = booster.get_score(importance_type='total_gain')
    feature_importance_df = pd.DataFrame(list(feature_importance_dict.items()), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.write('\n')
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('total_gain')
    plt.title('Feature Importance (importance_type="total_gain")')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    st.write('\n')
    st.markdown("**:blue[3-4. 전체 특성 상관관계 히트맵]**")
    st.write('\n')
    train_corr = train.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(train_corr, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

    st.write('\n')
    st.markdown("**:blue[3-5. 잔차 정규성]**")
    st.write('\n')
    train_residuals = train_target - train_pred
    plt.figure(figsize=(12, 8))
    stats.probplot(train_residuals, plot=plt)
    plt.title('QQ Plot')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.hist(train_residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    st.markdown("**:blue[3-6. 잔차 플롯]**")
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.scatter(train_pred, train_residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    st.pyplot(plt)



def lightgbm_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    st.write('\n')
    st.write('\n')


    st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
    feature_importance_df = pd.DataFrame({'Feature': best_model.feature_name_, 'Importance': best_model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.write('\n')
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('total_gain')
    plt.title('Feature Importance (importance_type="total_gain")')
    plt.xticks(rotation=90)
    st.pyplot(plt)

    
    st.write('\n')
    st.markdown("**:blue[3-4. 전체 특성 상관관계 히트맵]**")
    st.write('\n')
    train_corr = train.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(train_corr, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

    st.write('\n')
    st.markdown("**:blue[3-5. 잔차 정규성]**")
    st.write('\n')
    train_residuals = train_target - train_pred
    plt.figure(figsize=(12, 8))
    stats.probplot(train_residuals, plot=plt)
    plt.title('QQ Plot')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.hist(train_residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    st.markdown("**:blue[3-6. 잔차 플롯]**")
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.scatter(train_pred, train_residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    st.pyplot(plt)




       


def lightgbm_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    if val_pred is not None:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_name_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)
    


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_name_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')
            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)      

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")
            
    else:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_name_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')
            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')                


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_name_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")


def catboost_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    if val_pred is not None:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_names_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')
            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)
    


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_names_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            val_pred_probs = best_model.predict_proba(val)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            val_pred_probs_result = np.argmax(val_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            val_confusion_matrix = confusion_matrix(val_target, val_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            st.markdown("**:blue[4-3. Val 데이터 성능 평과 결과]**")
            val_report = classification_report(val_target, val_pred, output_dict=True)
            val_report_df = pd.DataFrame(val_report)
            st.write(val_report_df)      

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")
            
    else:
        if len(set(train_target)) == 2:
            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_names_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')

            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(train_target == classes[i], train_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))


            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6. Precision-Recall 커브]**")
            positive_probs = train_pred_probs[:, 1]

            # Precision-Recall Curve를 계산합니다.
            precision, recall, thresholds = precision_recall_curve(train_target, positive_probs)
            
            # AUC(Area Under the Curve) 값을 계산합니다.
            pr_auc = auc(recall, precision)
            
            # Precision-Recall Curve를 그립니다.
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='b', label='Precision-Recall curve (AUC = {:.2f})'.format(pr_auc))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off')
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')                


   
            
            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()                           
            
        # 다중 분류일 경우
        elif len(set(train_target)) > 2:

            classes = best_model.classes_
            st.write('\n')
            st.write('\n')


            st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
            feature_importance_df = pd.DataFrame({'Feature': best_model.feature_names_, 'Importance': best_model.feature_importances_})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            
            st.write('\n')
            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Features')
            plt.ylabel('total_gain')
            plt.title('Feature Importance (importance_type="total_gain")')
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.write('\n')
            st.write('\n')
            st.markdown("**:blue[3-4-1. 전체 클래스 특성 상관관계 히트맵]**")
            st.write('\n')
            train_corr = train.corr()
    
            plt.figure(figsize=(12, 8))
            sns.heatmap(train_corr, cmap='coolwarm')
            plt.title('Feature Correlation Heatmap')
            st.pyplot(plt)

            train_pred_probs = best_model.predict_proba(train)
            test_pred_probs = best_model.predict_proba(test)

            train_pred_probs_result = np.argmax(train_pred_probs, axis=1)
            test_pred_probs_result = np.argmax(test_pred_probs, axis=1)

            train_confusion_matrix = confusion_matrix(train_target, train_pred)
            test_confusion_matrix = confusion_matrix(test_target, test_pred)

            st.write('\n')
            st.markdown("**:blue[3-4-2. 전체 클래스 혼동행렬]**")
            plt.figure(figsize=(8.54, 3.46))
            sns.heatmap(train_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-4-3. 개별 클래스 혼동행렬]**")
            st.write('\n')
            select_class = st.selectbox("클래스를 선택하세요", classes)
            cm = confusion_matrix(train_target == select_class, train_pred == select_class)
            plt.figure(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Class: {select_class}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(test_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Test Confusion Matrix')
            #st.pyplot(plt)

            #plt.figure(figsize=(8.54, 3.46))
            #sns.heatmap(val_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            #plt.xlabel('Predicted Labels')
            #plt.ylabel('True Labels')
            #plt.title('Va Confusion Matrix')
            #st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-5. ROC 커브]**")
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(select_class, roc_auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Plot')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.markdown("**:blue[3-6. Precision Recall 커브]**")
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(train_target == select_class, train_pred_probs[:, np.where(classes == select_class)[0][0]])
            plt.plot(recall, precision, label='Precision-Recall curve (class: {})'.format(select_class))
            plt.xlabel('Recall ')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower right')
            plt.grid()
            st.pyplot(plt)


            st.write('\n')
            st.markdown("**:blue[3-6-2. Precision Recall Thresholds 커브]**")

            plt.figure(figsize=(8, 6))
            class_idx = np.where(classes == select_class)[0][0]
            y_true_positive = train_target == select_class
            y_true_negative = train_target != select_class

            # 선택한 클래스의 확률값만 사용하여 Precision-Recall curve 계산
            precision, recall, thresholds = precision_recall_curve(np.concatenate((y_true_positive, y_true_negative)), 
                                                                    np.concatenate((train_pred_probs[:, class_idx], 
                                                                                    1 - train_pred_probs[:, class_idx])))

            plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
            plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Precision-Recall Trade-off - Class: {}'.format(select_class))
            plt.legend(loc='lower left')
            plt.grid()
            st.pyplot(plt)

            st.write('\n')
            st.write('\n')
            st.subheader("4. 분류 결과 성능 리포트")
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-1. Train 데이터 성능 평과 결과]**")
            train_report = classification_report(train_target, train_pred, output_dict=True)
            train_report_df = pd.DataFrame(train_report)
            st.write(train_report_df)
            st.write('\n')
            st.write('\n')

            st.markdown("**:blue[4-2. Test 데이터 성능 평과 결과]**")
            test_report = classification_report(test_target, test_pred, output_dict=True)
            test_report_df = pd.DataFrame(test_report)
            st.write(test_report_df)
            st.write('\n')
            st.write('\n')            

            # val 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(val_target == classes[i], val_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Validation Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
            #
            ## test 데이터에 대한 ROC Curve
            #plt.figure(figsize=(10, 6))
            #for i in range(len(classes)):
            #    fpr, tpr, _ = roc_curve(test_target == classes[i], test_pred_probs[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, label='ROC curve (class: {}) (AUC = {:.2f})'.format(classes[i], roc_auc))
            #
            #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('False Positive Rate')
            #plt.ylabel('True Positive Rate')
            #plt.title('ROC Curve - Test Data')
            #plt.legend(loc='lower right')
            #plt.grid()
            #plt.show()
        
        else:
            st.error("오류가 발생하였습니다. 분류 클래스 갯수를 확인해주세요.")

def catboost_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred):

    st.write('\n')
    st.write('\n')


    st.markdown("**:blue[3-3. 전체 클래스 특성 중요도]**")
    feature_importance_df = pd.DataFrame({'Feature': best_model.feature_names_, 'Importance': best_model.feature_importances_})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.write('\n')
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Features')
    plt.ylabel('total_gain')
    plt.title('Feature Importance (importance_type="total_gain")')
    plt.xticks(rotation=90)
    st.pyplot(plt)

    
    st.write('\n')
    st.markdown("**:blue[3-4. 전체 특성 상관관계 히트맵]**")
    st.write('\n')
    train_corr = train.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(train_corr, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

    st.write('\n')
    st.markdown("**:blue[3-5. 잔차 정규성]**")
    st.write('\n')
    train_residuals = train_target - train_pred
    plt.figure(figsize=(12, 8))
    stats.probplot(train_residuals, plot=plt)
    plt.title('QQ Plot')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.hist(train_residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Histogram')
    st.pyplot(plt)

    st.write('\n')
    st.write('\n')
    st.markdown("**:blue[3-6. 잔차 플롯]**")
    st.write('\n')
    plt.figure(figsize=(12, 8))
    plt.scatter(train_pred, train_residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    st.pyplot(plt)



def pred_data_download(train_pred, test_pred, val_pred, best_model):
    if val_pred is not None:
        st.write('\n')
        st.write('\n')
        train_pred = pd.DataFrame(train_pred)
        train_pred = train_pred.to_csv(index=False).encode('utf-8')

        test_pred = pd.DataFrame(test_pred)
        test_pred = test_pred.to_csv(index=False).encode('utf-8')
        
        val_pred = pd.DataFrame(val_pred)
        val_pred = val_pred.to_csv(index=False).encode('utf-8')

        joblib.dump(best_model, 'best_model.joblib')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(label="Train 예측데이터", data=train_pred, file_name='train_pred.csv',mime='text/csv')

        with col2:
            st.download_button(label="Test 예측데이터", data=train_pred, file_name='test_pred.csv',mime='text/csv')

        with col3:
            st.download_button(label="Val 예측데이터", data=train_pred, file_name='val_pred.csv',mime='text/csv')

        with col4:
            st.download_button(label="모델 데이터", data='best_model.joblib', file_name='best_model.joblib', mime='application/octet-stream')

    else:
        st.write('\n')
        st.write('\n')
        train_pred = pd.DataFrame(train_pred)
        train_pred = train_pred.to_csv(index=False).encode('utf-8')

        test_pred = pd.DataFrame(test_pred)
        test_pred = test_pred.to_csv(index=False).encode('utf-8')

        joblib.dump(best_model, 'best_model.joblib')

        col1, col2, col3  = st.columns(3)

        with col1:
            st.download_button(label="Train 예측데이터", data=train_pred, file_name='train_pred.csv',mime='text/csv')

        with col2:
            st.download_button(label="Test 예측데이터", data=train_pred, file_name='test_pred.csv',mime='text/csv')

        with col3:
            st.download_button(label="모델 데이터", data='best_model.joblib', file_name='best_model.joblib', mime='application/octet-stream')


def machine_learning_app():

    # 프로그램 시작부
    st.title("💻 XGBoost & LightGBM & CatBoost")
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.subheader("1. 데이터 업로드")
    try:
        sample_data, train, test, val, columns_train, columns_test, columns_val = file_upload()
    except Exception as e:
        print(e)

    st.markdown('----')
    st.subheader("2. 타겟값 분리")
    try:
        st.write('\n')
        train, test, val, train_target, test_target, val_target = drop_target(train, test, val, columns_train, columns_test, columns_val)
    except Exception as e:
        print(e)

    st.markdown('----')
    st.subheader("3. 머신러닝 알고리즘 선택")
    try:
        task, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred, model_select = algorithm_select(sample_data, train, test, val, train_target, test_target, val_target)

        if task == '분류':
            if model_select == "XGBoost":
                xgboost_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)

            elif model_select == "LightGBM":
                lightgbm_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)

            elif model_select == "CatBoost":
                catboost_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)

        elif task == '회귀':

            if model_select == "XGBoost":
                xgboost_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)

            elif model_select == "LightGBM":
                lightgbm_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)

            elif model_select == "CatBoost":
                catboost_regressor_algorithm_report(train, test, val, booster, best_model, train_target, test_target, val_target, train_pred, test_pred, val_pred)


    except Exception as e:
        st.error("오류 메세지" + "  " + str(e))  

    st.markdown('----')
    try:
        if task == '분류':
            st.subheader("5. 최종 예측완료 데이터 다운로드")
            pred_data_download(train_pred, test_pred, val_pred, best_model)
        else:
            st.subheader("4. 최종 예측완료 데이터 다운로드")
            pred_data_download(train_pred, test_pred, val_pred, best_model)


    except Exception as e:
        st.error("오류 메세지" + "  " + str(e))      

  

            