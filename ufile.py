import streamlit as st
import pandas as pd
import os

def save_uploaded_file(diretory,file) :
    #1. 디렉토리가 있는지 확인하여, 없으면 먼저, 디렉토리부터 만든다.
    if not os.path.exists(diretory):
        os.makedirs(diretory)
    
    #2. 디렉토리가 있으니, 파일을 저장한다.
    with open(os.path.join(diretory,file.name),'wb') as f:
         f.write(file.getbuffer())

    #3. 파일이 저장이 성공했으니, 화면에 성공했다고 보여주면서 리턴
    return st.success('{} 예 {} 파일이 저장되었습니다.'.format(diretory,file.name))