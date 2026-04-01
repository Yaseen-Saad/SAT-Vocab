from dotenv import load_dotenv
load_dotenv()
from src.services.llm_service import get_llm_service
from src.core.rag_engine_clean import get_rag_engine
from src.core.vocabulary_generator_clean import SimpleVocabularyGenerator

svc = get_llm_service()
print('provider=', svc.provider)
print('base_url=', svc.base_url)
print('model=', svc.default_model)
rag = get_rag_engine()
gen = SimpleVocabularyGenerator(svc, rag)
entry = gen.generate_entry('analogous')
print('quality=', entry.quality_score)
print('valid=', entry.validation_passed)
print('llm_error=', entry.llm_error)
print('definition=', entry.definition)
print('mnemonic=', entry.mnemonic_phrase)
print('sentence=', entry.example_sentence)
print('generated_text_has_content=', bool(entry.generated_text))
