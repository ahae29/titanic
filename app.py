import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 타이타닉 데이터셋 로드
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

titanic = load_data()

# 필수 피처 선택 및 결측치 처리
titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()

# 성별 인코딩
label_encoder = LabelEncoder()
titanic['Sex'] = label_encoder.fit_transform(titanic['Sex'])

# feature와 target 설정
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic['Survived']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 웹앱 타이틀
st.title("타이타닉 생존 예측기")

# 사용자 입력 받기
pclass = st.selectbox("객실 등급 (1, 2, 3)", [1, 2, 3])
sex = st.selectbox("성별", ["남성", "여성"])
age = st.number_input("나이", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("형제/자매 수", min_value=0, max_value=10, value=0)
parch = st.number_input("부모/자녀 수", min_value=0, max_value=10, value=0)
fare = st.number_input("요금", min_value=0.0, value=10.0)

# 성별 인코딩
sex_encoded = label_encoder.transform([sex])[0]

# 예측 버튼
if st.button("예측하기"):
    input_data = [[pclass, sex_encoded, age, sibsp, parch, fare]]
    prediction = model.predict(input_data)
    st.success(f"예측된 생존 여부: {'생존' if prediction[0] == 1 else '사망'}")

# 앱 실행 안내
st.write("위 입력값을 바탕으로 타이타닉 생존 여부를 예측합니다.")
