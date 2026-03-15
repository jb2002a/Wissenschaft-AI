import json
import re
import dspy

from src.translation.modules.invoke import invoke

def normalize_ko(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

# 1) 데이터 로드
with open("resources/initial_test_json_file.json", "r", encoding="utf-8") as f:
    row = json.load(f)

# 2) DSPy Example로 감싸기 (입력 필드 지정)
example = dspy.Example(
    original_text=row["original_text"],
    translated_text=row["translated_text"],
).with_inputs("original_text")

# 3) invoke()를 DSPy Prediction 형태로 래핑
pred_text = invoke(example.original_text)
pred = dspy.Prediction(translated_text=pred_text)

# 4) metric 정의 (가장 단순: 정규화 후 exact match)
def metric_exact(example, pred, trace=None):
    gold = normalize_ko(example.translated_text)
    guess = normalize_ko(pred.translated_text)
    return 1.0 if gold == guess else 0.0

score = metric_exact(example, pred)
print("pred:", pred.translated_text)
print("gold:", example.translated_text)
print("score:", score)