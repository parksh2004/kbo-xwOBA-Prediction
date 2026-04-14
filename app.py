import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="KBO xwOBA 예측 대시보드", layout="centered")

@st.cache_resource
def load_model():
    # 저장된 패키지(딕셔너리)를 불러옵니다.
    return joblib.load('xwOBA_ridge_model.pkl')

try:
    model_package = load_model()
    scaler = model_package['scaler'] # 스케일러 분리
    model = model_package['model']   # 모델 분리
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

st.title("⚾ KBO 투수 타구 질(xwOBA) 예측 대시보드")
st.markdown("선수의 주요 스탯을 입력하면 모델이 추정하는 xwOBA를 계산합니다.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🎯 핵심 제어 지표 (%)**")
    k_rate = st.slider("삼진 비율 (K_rate)", 0.0, 40.0, 22.0) / 100
    bb_rate = st.slider("볼넷 비율 (BB_rate)", 0.0, 20.0, 8.0) / 100
    swstr_rate = st.slider("헛스윙 비율 (SwStr_rate)", 0.0, 25.0, 11.0) / 100
    hr_rate = st.slider("피홈런 비율 (HR_rate_neutral)", 0.0, 10.0, 3.0) / 100

with col2:
    st.markdown("**🏟️ 타구 프로필 (%)**")
    gb_rate = st.slider("땅볼 비율 (GB_rate_neutral)", 0.0, 80.0, 43.0) / 100
    fb_rate = st.slider("플라이볼 비율 (FB_rate_neutral)", 0.0, 60.0, 36.0) / 100
    ld_rate = st.slider("라인드라이브 비율 (LD_rate_neutral)", 0.0, 50.0, 21.0) / 100
    pop_rate = st.slider("팝업 비율 (POP_rate_neutral)", 0.0, 20.0, 6.0) / 100

with st.expander("세부 카운트 지표 (선택 사항)"):
    c1, c2, c3 = st.columns(3)
    mean_balls = c1.number_input("인플레이 직전 평균 볼", value=1.2)
    mean_strikes = c2.number_input("인플레이 직전 평균 스트라이크", value=0.9)
    mean_fouls = c3.number_input("인플레이 직전 평균 파울", value=0.5)

input_data = pd.DataFrame({
    "K_rate": [k_rate],
    "BB_rate": [bb_rate],
    "HR_rate_neutral": [hr_rate],
    "GB_rate_neutral": [gb_rate],
    "FB_rate_neutral": [fb_rate],
    "LD_rate_neutral": [ld_rate],
    "POP_rate_neutral": [pop_rate],
    "mean_balls_before": [mean_balls],
    "mean_strikes_before": [mean_strikes],
    "mean_fouls_before": [mean_fouls],
    "SwStr_rate": [swstr_rate]
})

if st.button("🚀 xwOBA 예측하기", use_container_width=True):
    # 입력된 원본 데이터를 스케일러로 먼저 변환한 뒤 예측 수행!
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    st.divider()
    
    if prediction < 0.290:
        eval_text, color = "✨ 특급 (Excellent)", "#1f77b4"
    elif prediction < 0.320:
        eval_text, color = "👍 우수 (Good)", "#2ca02c"
    elif prediction < 0.340:
        eval_text, color = "🤔 평균 수준 (Average)", "#ff7f0e"
    else:
        eval_text, color = "⚠️ 위험 (Poor)", "#d62728"
        
    st.markdown(f"<h1 style='text-align: center; color: {color};'>예측 xwOBA: {prediction:.3f}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>평가: {eval_text}</h3>", unsafe_allow_html=True)
