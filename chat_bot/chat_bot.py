import requests
import streamlit as st
from bardapi.constants import SESSION_HEADERS
from bardapi import Bard
from bardapi import BardCookies
import bardapi
import os



def chat_bot():
    
    API_KEY = st.secrets["__Secure-1PSID"][0]
    API_KEY2 = st.secrets["__Secure-1PSIDTS"][0]
    API_KEY3 = st.secrets["__Secure-1PSIDCC"][0]
    

    cookie_dict = {"__Secure-1PSID":API_KEY, "__Secure-1PSIDTS":API_KEY2, "__Secure-1PSIDCC":API_KEY3}

    #session = requests.Session()
    #session.headers = SESSION_HEADERS
    #session.cookies.set("__Secure-1PSID", API_KEY) 

    session = requests.Session()
    session.headers = SESSION_HEADERS
    session.cookies.set("__Secure-1PSID", API_KEY)
    session.cookies.set("__Secure-1PSIDTS", API_KEY2)
    session.cookies.set("__Secure-1PSIDCC", API_KEY3)
    # many models use triple hash '###' for keywords, Vicunas are simpler:
    
    st.title("🤖 Chat-bot")
    
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

    
    st.info("해당 챗봇은 Google Bard API를 사용하였으며,  Google 정책에 따라 서비스가 종료될 수 있습니다.")
    st.warning("해당 챗봇은 테스트용으로 이전대화를 기록하지 않습니다.")
    if "message" not in st.session_state:
        st.session_state.message = []

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    
    # Handle previous messages
    for message in st.session_state.message:
        with st.chat_message(message.get("role")):
            st.write(message.get('content'))
            
    

    prompt = st.chat_input("궁금하신게 무엇입니까?")


    #if st.session_state.conversation_id is None:
    #    a = "안녕"
    #    with st.chat_message("user"):
    #        st.write(a)

    #    response1 = bardapi.core.Bard(API_KEY).get_answer(a)
    #    response1_content = response1['choices'][0]['content'][0]
    #    st.session_state.message.append({"role": 'user', 'content': a})
    #    st.session_state.message.append({"role": 'assistant', 'content': response1_content})
    #    st.session_state.conversation_id = response1['conversation_id']
    #    with st.chat_message("assitant"):
    #        st.write(response1_content)
      


    if prompt:
        prompt = str(prompt)
        
        st.session_state.message.append({'role': "user", 'content': prompt})
        with st.chat_message("user"):
            st.write(prompt)

        bard = BardCookies(cookie_dict=cookie_dict, session=session, conversation_id=st.session_state.conversation_id)
        response = bard.get_answer(prompt)['choices'][0]['content'][0]

        st.session_state.message.append({"role": 'assistant', 'content': response})

        with st.chat_message('assistant'):
            st.write(response, unsafe_allow_html=True)
