"""invoke 호출용 Signature: 사용자 프롬프트 → JSON 형식 응답."""

import dspy


class Invoke(dspy.Signature):
    """사용자 요청에 따라 응답한다. 응답은 반드시 JSON 형식의 문자열만 출력한다."""

    prompt: str = dspy.InputField(desc="사용자 요청 또는 지시")
    response: str = dspy.OutputField(
        desc='JSON 형식 문자열. 예: {"text": "응답 내용", "lang": "ko"}. text는 필수.'
    )
