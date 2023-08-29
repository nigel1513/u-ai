from st_on_hover_tabs import on_hover_tabs
import streamlit as st
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
from contact import contact
from statistics1 import statistics1
from preprocessing import preprocessing
from  Machine_Learning import machine_learning
from text_cloud import text
from Home import home
from chat_bot import chat_bot

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


def main():
    st.set_page_config()

    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Main', 'Statistics', 'Preprocessing', 'Machine Learning', 'Chat-Bot', 'WordCloud','Contact'], 
                             iconName=['home', 'bar_chart', 'build', 'smart_toy', 'chat', 'filter_drama','contact_support'], default_choice=0)

        

    if tabs =='Main':
        home.home_app()
        

    elif tabs == 'Statistics':
        statistics1.statistics_app()

    elif tabs == 'Preprocessing':
        preprocessing.preprocessing_app()

    elif tabs == 'Machine Learning':
        machine_learning.machine_learning_app()

    elif tabs == 'WordCloud':
        text.text_app()

    elif tabs == 'Contact':
        contact.contact()

    elif tabs == 'Chat-Bot':
        chat_bot.chat_bot()

    else:
        pass

if __name__ == '__main__':
    main()
