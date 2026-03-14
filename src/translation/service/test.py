from src.common.models.gemini import get_gemini_31_pro_llm
from src.translation.prompt.translation_prompt import get_translation_prompt_template

llm = get_gemini_31_pro_llm()
prompt_template = get_translation_prompt_template()

llm_chain = prompt_template | llm

print(llm_chain.invoke({"text": "Hello, world!"}))