from pathlib import Path
import streamlit as st
import pandas as pd
from wordcloud import STOPWORDS, WordCloud
from matplotlib import font_manager, rc
import seaborn as sns
from konlpy.tag import Okt
import matplotlib as mplc
import numpy as np
import pandas as pd
import io
import re
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import kss


# 파일업로드 함수
 # Set cache expiration time (seconds)
def load_dataframe(upload_file):
    try:
        if upload_file.name.endswith('.csv'):
            df = pd.read_csv(upload_file, encoding='utf-8')
        elif upload_file.name.endswith('.xlsx'):
            df = pd.read_excel(upload_file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

    columns = list(df.columns)
    columns.insert(0, None)
    return df

# 전처리 함수
def preprocess_korean_text(text):
    # HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)

    # 숫자, 특수문자 등 필요하지 않은 언어 제거 (영어와 한글은 유지)
    text = re.sub(r'[^\w\s가-힣.!?]', '', text)

    # 여러 개의 공백을 하나의 공백으로 변경
    text = re.sub(r"\s+", ' ', text)

    #text = text.replace(" ","")

    # 문장 앞뒤의 공백 제거
    text = text.strip()
    return text


# 형태소 분석 및 품사 추출 함수 정의
#def analyze_morphemes(text):
#    okt = Okt()
#    morphemes = okt.pos(text)
#    nouns_only = [morpheme for morpheme, pos in morphemes if pos == 'Noun']
#    verbs_only = [morpheme for morpheme, pos in morphemes if pos == 'Verb']
#    adjectives_only = [morpheme for morpheme, pos in morphemes if pos == 'Adjective']
    #adverb_only = [morpheme for morpheme, pos in morphemes if pos == 'Adverb']
    #determiner_only = [morpheme for morpheme, pos in morphemes if pos == 'Determiner']
    #conjunction_only = [morpheme for morpheme, pos in morphemes if pos == 'Conjunction']
    #exclamation_only = [morpheme for morpheme, pos in morphemes if pos == 'Exclamation']
    #josa_only = [morpheme for morpheme, pos in morphemes if pos == 'Josa']
    #eomi_only = [morpheme for morpheme, pos in morphemes if pos == 'Eomi']
    #preposition_only = [morpheme for morpheme, pos in morphemes if pos == 'Preposition']
    #prefix_only = [morpheme for morpheme, pos in morphemes if pos == 'Prefix']
    #suffix_only = [morpheme for morpheme, pos in morphemes if pos == 'Suffix']
    #foreign_only = [morpheme for morpheme, pos in morphemes if pos == 'Foreign']
    #number_only = [morpheme for morpheme, pos in morphemes if pos == 'Number']
    #unknown_only = [morpheme for morpheme, pos in morphemes if pos == 'Unknown']
    #alpha_only = [morpheme for morpheme, pos in morphemes if pos == 'Alpha']
    #prectuation_only = [morpheme for morpheme, pos in morphemes if pos == 'Punctuation']


    #return nouns_only, verbs_only, adjectives_only
    #adverb_only, determiner_only, conjunction_only, exclamation_only, josa_only, eomi_only, preposition_only, prefix_only, suffix_only, foreign_only, number_only, unknown_only, alpha_only, prectuation_only

def analyze_morphemes(text):
    okt = Okt()
    morphemes = okt.pos(text)
    morpheme_dict = {}
    for morpheme, pos in morphemes:
        if pos in morpheme_dict:
            morpheme_dict[pos].append(morpheme)
        else:
            morpheme_dict[pos] = [morpheme]
    return morpheme_dict

def text_app():
    font_path = "sample_data/malgun.ttf"
    font_name = font_manager.FontProperties(fname="sample_data/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

    st.title("⛅ 워드클라우드 & 문장 분석")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

    st.subheader("1. 파일 업로드")
    st.write("\n")

    uploaded_files = st.file_uploader("CSV 파일을 업로드해주세요.", type=['xlsx', 'csv'])

    if uploaded_files is not None:    

        font_path = "sample_data/malgun.ttf"
        font_name = font_manager.FontProperties(fname="sample_data/malgun.ttf").get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False

        style_image_path = 'sample_data/asdf.png'
        style_image = Image.open(style_image_path)
        style_image_array = np.array(style_image)
        df = load_dataframe(uploaded_files)

        #df = pd.read_excel(uploaded_files, engine='openpyxl')
        st.success('파일업로드 완료', icon="🔥")
        st.markdown("---")
        st.write("\n")

        st.subheader("2. 전체 데이터 확인하기")
        st.write("\n")
        st.write("\n")
        st.dataframe(df, width=1200)
        st.markdown("---")
        st.write("\n")
        st.subheader("3. 컬럼 선택")
        st.write("\n")
        df_select = st.selectbox("분석하고 싶은 컬럼을 선택하세요.", df.select_dtypes(include=['object']).columns.to_list(), help="문자형 컬럼만 목록에 노출됩니다.")
        st.info("HTML 태그, 숫자, 특수문자, 공백 제거, 문장 분리가 자동으로 이뤄집니다.", icon="🎉")
        st.write("\n")
        st.write("\n")
        if len(df.select_dtypes(include=['object']).columns.to_list()) >= 1:
            st.markdown("**:blue[3-1. 변경된 데이터 확인]**")
            select_column = df[df_select].dropna()
            select_column_preprocessed = select_column.apply(preprocess_korean_text)

            sentences_per_row = select_column_preprocessed.apply(kss.split_sentences)
            sentences_per_row_flattened = [sentence for sentences in sentences_per_row for sentence in sentences]
            new_df = pd.DataFrame({'sentence': sentences_per_row_flattened})
            st.dataframe(new_df, width=1200)

            st.markdown("---")
            st.write("\n")
            st.subheader("4. 형태소 분석")
            st.write("\n")

            all_morphemes_dict = {}
            for idx, sentence in enumerate(new_df['sentence'].values):
                morpheme_dict = analyze_morphemes(sentence)
                for pos, morphemes in morpheme_dict.items():
                    if pos in all_morphemes_dict:
                        all_morphemes_dict[pos].extend(morphemes)
                    else:
                        all_morphemes_dict[pos] = morphemes

            max_length = max(len(morphemes) for morphemes in all_morphemes_dict.values())
            for pos, morphemes in all_morphemes_dict.items():
                if len(morphemes) < max_length:
                    all_morphemes_dict[pos].extend([''] * (max_length - len(morphemes)))
            morphemes_df = pd.DataFrame(all_morphemes_dict)
            st.dataframe(morphemes_df, width=1200)

            st.markdown("---")
            st.subheader("5. 품사별 빈도 시각화")
            st.write("\n")
            pos_columns = morphemes_df.columns[1:]

            # 품사별 빈도수 계산
            pos_counts = {}
            for pos in pos_columns:
                pos_counts[pos] = morphemes_df[pos].explode().value_counts()


            for pos, counts in pos_counts.items():
                plt.figure(figsize=(10, 6))
                counts = counts[counts.index != '']  # 빈값 제외
                top_counts = counts[:10]  # 상위 10개만 선택
                top_counts.plot(kind='bar')
                plt.title(f'Top 10 {pos} 빈도수', fontsize=16)
                plt.xlabel('형태소', fontsize=14)
                plt.ylabel('빈도수', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                st.write('\n')

            st.markdown("---")
            st.subheader("6. 명사 워드클라우드")
            st.write("\n")

            nouns = morphemes_df['Noun'].explode().dropna().tolist()
            noun_counts = pd.Series(nouns).value_counts()

            #verbs = morphemes_df['Verb'].explode().dropna().tolist()
            #verb_counts = pd.Series(verbs).value_counts()

            #adjective = morphemes_df['Adjective'].explode().dropna().tolist()
            #adjective_counts = pd.Series(adjective).value_counts()

            # 워드클라우드 생성
            noun_wordcloud = WordCloud(max_font_size=150, font_path="sample_data/malgun.ttf", background_color="white", width=800,max_words=150,
            height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(noun_counts)


            # 워드클라우드 그리기
            plt.figure(figsize=(10, 6))
            plt.imshow(noun_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt.gcf())
            st.write('\n')
            noun_wordcloud.to_file("noun_wordcloud.png")
            st.download_button("Download WordCloud Image", "noun_wordcloud.png")

            #verbs_wordcloud = WordCloud(max_font_size=150, font_path="text_cloud/malgun.ttf", background_color="white", width=800,max_words=150,
            #height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(verb_counts)

            # 워드클라우드 그리기
            # plt.figure(figsize=(10, 6))
            # plt.imshow(verbs_wordcloud, interpolation='bilinear')
            # plt.title('동사 워드클라우드')
            # plt.axis('off')
            # st.pyplot(plt.gcf())
            # st.write('\n')
            # verbs_wordcloud.to_file("verbs_wordcloud.png")
            # st.download_button("Download WordCloud Image", "verbs_wordcloud.png")

            #adjective_wordcloud = WordCloud(max_font_size=150, font_path="text_cloud/malgun.ttf", background_color="white", width=800,max_words=150,
            #height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(adjective_counts)
            #
            ## 워드클라우드 그리기
            #plt.figure(figsize=(10, 6))
            #plt.imshow(adjective_wordcloud, interpolation='bilinear')
            #plt.title('형용사 워드클라우드')
            #plt.axis('off')
            #st.pyplot(plt.gcf())
            #st.write('\n')
            #adjective_wordcloud.to_file("adjective_wordcloud.png")
            #st.download_button("Download WordCloud Image", "adjective_wordcloud.png")

            # 결과 출력
        else:
            st.error("문자형 컬럼이 없습니다.")

    

    else:
        sample_data = st.checkbox('샘플데이터 사용', value=True)
        if sample_data:
            font_path = "sample_data/malgun.ttf"
            font_name = font_manager.FontProperties(fname="sample_data/malgun.ttf").get_name()
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
            style_image_path = 'sample_data/asdf.png'
            style_image = Image.open(style_image_path)
            style_image_array = np.array(style_image)
            df = pd.read_excel('sample_data/asdf1.xlsx', engine='openpyxl')

            #df = pd.read_excel(uploaded_files, engine='openpyxl')
            st.success('파일업로드 완료', icon="🔥")
            st.markdown("---")
            st.write("\n")

            st.subheader("2. 전체 데이터 확인하기")
            st.write("\n")
            st.write("\n")
            st.dataframe(df, width=1200)
            st.markdown("---")
            st.write("\n")
            st.subheader("3. 컬럼 선택")
            st.write("\n")
            df_select = st.selectbox("분석하고 싶은 컬럼을 선택하세요.", df.select_dtypes(include=['object']).columns.to_list(), help="문자형 컬럼만 목록에 노출됩니다.")
            st.info("HTML 태그, 숫자, 특수문자, 공백 제거, 문장 분리가 자동으로 이뤄집니다.", icon="🎉")
            st.write("\n")
            st.write("\n")
            if len(df.select_dtypes(include=['object']).columns.to_list()) >= 1:
                st.markdown("**:blue[3-1. 변경된 데이터 확인]**")
                select_column = df[df_select].dropna()
                select_column_preprocessed = select_column.apply(preprocess_korean_text)

                sentences_per_row = select_column_preprocessed.apply(kss.split_sentences)
                sentences_per_row_flattened = [sentence for sentences in sentences_per_row for sentence in sentences]
                new_df = pd.DataFrame({'sentence': sentences_per_row_flattened})
                st.dataframe(new_df, width=1200)

                st.markdown("---")
                st.write("\n")
                st.subheader("4. 형태소 분석")
                st.write("\n")

                all_morphemes_dict = {}
                for idx, sentence in enumerate(new_df['sentence'].values):
                    morpheme_dict = analyze_morphemes(sentence)
                    for pos, morphemes in morpheme_dict.items():
                        if pos in all_morphemes_dict:
                            all_morphemes_dict[pos].extend(morphemes)
                        else:
                            all_morphemes_dict[pos] = morphemes

                max_length = max(len(morphemes) for morphemes in all_morphemes_dict.values())
                for pos, morphemes in all_morphemes_dict.items():
                    if len(morphemes) < max_length:
                        all_morphemes_dict[pos].extend([''] * (max_length - len(morphemes)))
                morphemes_df = pd.DataFrame(all_morphemes_dict)
                st.dataframe(morphemes_df, width=1200)

                st.markdown("---")
                st.subheader("5. 품사별 빈도 시각화")
                st.write("\n")
                pos_columns = morphemes_df.columns[1:]

                # 품사별 빈도수 계산
                pos_counts = {}
                for pos in pos_columns:
                    pos_counts[pos] = morphemes_df[pos].explode().value_counts()


                for pos, counts in pos_counts.items():
                    plt.figure(figsize=(10, 6))
                    counts = counts[counts.index != '']  # 빈값 제외
                    top_counts = counts[:10]  # 상위 10개만 선택
                    top_counts.plot(kind='bar')
                    plt.title(f'Top 10 {pos} 빈도수', fontsize=16)
                    plt.xlabel('형태소', fontsize=14)
                    plt.ylabel('빈도수', fontsize=14)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    st.write('\n')

                st.markdown("---")
                st.subheader("6. 명사 워드클라우드")
                st.write("\n")

                nouns = morphemes_df['Noun'].explode().dropna().tolist()
                noun_counts = pd.Series(nouns).value_counts()

                #verbs = morphemes_df['Verb'].explode().dropna().tolist()
                #verb_counts = pd.Series(verbs).value_counts()

                #adjective = morphemes_df['Adjective'].explode().dropna().tolist()
                #adjective_counts = pd.Series(adjective).value_counts()

                # 워드클라우드 생성
                noun_wordcloud = WordCloud(max_font_size=150, font_path="sample_data/malgun.TTF", background_color="white", width=800,max_words=150,
                height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(noun_counts)


                # 워드클라우드 그리기
                plt.figure(figsize=(10, 6))
                plt.imshow(noun_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt.gcf())
                st.write('\n')
                noun_wordcloud.to_file("noun_wordcloud.png")
                st.download_button("Download WordCloud Image", "noun_wordcloud.png")

                #verbs_wordcloud = WordCloud(max_font_size=150, font_path="text_cloud/malgun.TTF", background_color="white", width=800,max_words=150,
                #height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(verb_counts)

                # 워드클라우드 그리기
                # plt.figure(figsize=(10, 6))
                # plt.imshow(verbs_wordcloud, interpolation='bilinear')
                # plt.title('동사 워드클라우드')
                # plt.axis('off')
                # st.pyplot(plt.gcf())
                # st.write('\n')
                # verbs_wordcloud.to_file("verbs_wordcloud.png")
                # st.download_button("Download WordCloud Image", "verbs_wordcloud.png")

                #adjective_wordcloud = WordCloud(max_font_size=150, font_path="text_cloud/malgun.TTF", background_color="white", width=800,max_words=150,
                #height=400, mask=style_image_array, colormap='magma', relative_scaling=0.5).generate_from_frequencies(adjective_counts)
                #
                ## 워드클라우드 그리기
                #plt.figure(figsize=(10, 6))
                #plt.imshow(adjective_wordcloud, interpolation='bilinear')
                #plt.title('형용사 워드클라우드')
                #plt.axis('off')
                #st.pyplot(plt.gcf())
                #st.write('\n')
                #adjective_wordcloud.to_file("adjective_wordcloud.png")
                #st.download_button("Download WordCloud Image", "adjective_wordcloud.png")

                # 결과 출력
            else:
                st.error("문자형 컬럼이 없습니다.")
        else:
            st.info("파일을 업로드해주세요")
    st.write("\n")
    st.write("\n")
    st.write("\n")



