"""번역용 시스템 프롬프트 및 LCEL용 Runnable 프롬프트."""

from langchain_core.prompts import ChatPromptTemplate


def get_translation_prompt() -> str:
    """번역용 시스템 프롬프트 문자열을 반환합니다."""
    return """
Role: You are a professional academic translator specializing in German philosophy and conceptual history (Begriffsgeschichte).

Task: Translate the provided German philosophical text into Korean.

Requirements:

Terminology: Use precise Korean academic terms established in philosophical discourses (e.g., Kantian, Hegelian, or Phenomenological terminology).

Tone: Maintain a formal, objective, and scholarly prose style (Haera-che).

Accuracy: Preserving the exact logical structure and conceptual nuances of the original German is the highest priority.

Clarity: Ensure that complex German sentence structures are rendered into clear, readable academic Korean without losing technical rigor.
""".strip()


def get_translation_prompt_template() -> ChatPromptTemplate:
    """LCEL에서 사용할 수 있는 Runnable 프롬프트 템플릿을 반환합니다.

    입력: {"text": "번역할 독일어 문장"}
    체인 예: get_translation_prompt_template() | llm | StrOutputParser()
    """
    return ChatPromptTemplate.from_messages([
        ("system", get_translation_prompt()),
        ("human", "{text}"),
    ])