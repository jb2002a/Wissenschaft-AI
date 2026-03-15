"""LM 로드(get_lm) 및 predict(invoke)."""

import os

import dspy
from dotenv import load_dotenv

from src.translation.signatures.german_to_korean import GermanToKorean

load_dotenv()

_DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"

def get_lm() -> None:
    """DSPy LM을 로드하고 전역으로 설정한다. 기본: gemini/gemini-3.1-pro-preview."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
    lm = dspy.LM(_DEFAULT_MODEL, api_key=api_key, temperature=0.0)
    dspy.configure(lm=lm)


def invoke(original_text: str) -> str:
    """프롬프트를 LM에 보내고 predict한 응답 문자열을 반환한다."""
    
    get_lm()

    predictor = dspy.Predict(GermanToKorean)

    out = predictor(original_text=original_text)
    return out.translated_text


def show_last_prompt(n: int = 1) -> None:
    """마지막 n회 호출의 자동 생성 프롬프트와 응답을 콘솔에 출력한다.
    invoke() 호출 후에 호출하면 된다."""
    if dspy.settings.lm is None:
        raise RuntimeError("LM이 아직 로드되지 않았습니다. get_lm() 후 invoke()를 먼저 호출하세요.")
    dspy.settings.lm.inspect_history(n=n)


# __main__ for testing
if __name__ == "__main__":
    result = invoke(
        """
        Dächte man sich rein geistige Wesen in einem aus solchen allein bestehenden Personenreich, so würde ihr Hervortreten, ihre Erhaltung und Entwicklung, wie ihr Verschwinden (welche Vorstellungen man auch von dem Hintergrund sich bilde, aus welchem sie hervorträten und in den sie wieder zurücktreten würden), an Bedingungen geistiger Art gebunden sein; ihr Wohlsein wäre in ihrer Lage zur geistigen Welt gegründet; ihre Verbindung untereinander, ihre Handlungen aufeinander würden sich durch rein geistige Mittel vollziehen und die dauernden Wirkungen ihrer Handlungen würden rein geistiger Art sein; selbst ihr Zurücktreten aus dem Reich der Personen würde in dem Geistigen seinen Grund haben. Das System solcher Individuen würde in reinen Geisteswissenschaften erkannt werden. In Wirklichkeit entsteht ein Individuum, wird erhalten und entwickelt sich auf Grund der Funktionen des tierischen Organismus und ihrer Beziehungen zu dem umgebenden Naturlauf; sein Lebensgefühl ist wenigstens teilweise in diesen Funktionen gegründet; seine Eindrücke sind von den Sinnesorganen und ihren Affektionen seitens der Außenwelt bedingt; den Reichtum und die Beweglichkeit seiner Vorstellungen und die Stärke sowie die Richtung seiner Willensakte finden wir vielfach von Veränderungen in seinem Nervensystem abhängig. Sein Willensantrieb bringt Muskelfasern zur Verkürzung, und so ist .ein Wirken nach außen an Veränderungen in den Lageverhältnissen der Massenteilchen des Organismus gebunden; dauernde Erfolge seiner Willens handlunger existieren nur in der Form von Veränderungen innerhalb der materiellen Welt. So ist das geistige Leben eines Menschen ein nur durch Abstraktion loslösbarer Teil der psycho-physischen Lebenseinheit, als welche ein Menschendasein und Menschenleben sich darstellt. Das System dieser Lebenseinheiten ist die Wirklichkeit, welche den Gegenstand der geschichtlich-gesellschaftlichen Wissenschaften ausmacht.
        """
    )
    print(result)
