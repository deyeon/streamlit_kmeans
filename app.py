import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_object_dtype
from pandas.api.types import is_numeric_dtype
from ufile import save_uploaded_file
from datetime import date,datetime
from PIL import Image

# 웹 대시보드 프레임워크인, 스트림릿은
# main 함수가 있어야 한다.

def main() :
    st.title ('K-Means클러스터링')

    #1. csv파일을 업로드 할수 있다.

    
    st.subheader('CSV 파일 업로드')

    file = st.file_uploader('CSV파일업로드',type=['csv'])
        
    if file is not None:

        save_uploaded_file('csv',file)

        #csv 파일은, 판다스로 읽어서 화면에 보여준다.

        df = pd.read_csv(file)
        df = df.dropna(axis=0)
        st.dataframe(df)

        column_list = df.columns
        
    #3. KMeams 클러스터링을  하기 위해 x로 사용할 컬럼을 설정 할 수있다.

        selected_columns=st.multiselect('X로 사용할 컬럼을 선택하세요',df.columns)
        

        if len(selected_columns) !=0:
            
            if is_numeric_dtype(df[selected_columns])==True:

                X= df[selected_columns]

                st.dataframe(X)

        #4. WCSS를 확인하기 위한 그룹의 갯수를 정할수 있다.1~10개

                st.subheader('WCSS를 위한 클러스터링 갯수를 선택')

                max_number = st.slider('최대 그룹 선택',2,10,value=10)

                from sklearn.cluster import KMeans

                wcss = []
                for k in np.arange(1,max_number+1):
                    kmeans = KMeans(n_clusters=k ,random_state=5)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)

            # st.write(wcss)
                fig1 = plt.figure()
                x = np.arange(1,max_number+1)    
                fig=plt.plot(x,wcss)
                plt.title('The Elbow Method')
                plt.xlabel('Number of Clusters')
                plt.ylabel('wcss')
                st.pyplot(fig1)

            #실제로 그룹핑할 갯수 선택!
                k = st.number_input('그룹 갯수 결정',1,max_number)

                kmeans = KMeans(n_clusters=k , random_state=5)

                y_pred=kmeans.fit_predict(X)

                df["Group"]=y_pred

                st.dataframe(df.sort_values('Group'))

                df.to_csv('result.csv')
            


    

    
if __name__ == '__main__' :
    main()