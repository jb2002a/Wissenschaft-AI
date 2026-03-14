import streamlit as st
from src.common.models.gemini import get_gemini_31_pro_llm
from src.translation.prompt.translation_prompt import get_translation_prompt_template

# 체인은 한 번만 구성 (캐싱하면 더 좋음)
@st.cache_resource
def get_chain():
    llm = get_gemini_31_pro_llm()
    prompt_template = get_translation_prompt_template()
    return prompt_template | llm

st.title("독일어 → 한국어 번역")

text = st.text_area(
    "번역할 독일어 텍스트",
    placeholder="독일어 문장을 입력하세요...",
    height=150,
)

if st.button("번역"):
    if not text.strip():
        st.warning("텍스트를 입력해 주세요.")
    else:
        with st.spinner("번역 중..."):
            chain = get_chain()
            result = chain.invoke({"text": text.strip()})
        st.success("번역 결과")
        st.write(result.content if hasattr(result, "content") else result)