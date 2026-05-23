"""
Verify that OSS and Frontier assistants use different model chains.
"""
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env')

from utils.hf_model import generate_response as hf_gen
from utils.openrouter_model import generate_response as or_gen

print('=' * 60)
print('VERIFICATION: OSS vs Frontier model usage')
print('=' * 60)

# OSS
r1 = hf_gen([{'role': 'user', 'content': 'Say hello in 5 words.'}])
print(f'\n[OSS] model_used: {r1["model_used"]}')
print(f'      error:      {r1["error"]}')

# Frontier
r2 = or_gen([{'role': 'user', 'content': 'Say hello in 5 words.'}])
print(f'\n[FRONTIER] model_used: {r2["model_used"]}')
print(f'           error:      {r2["error"]}')

print(f'\n{"=" * 60}')
print(f'OSS model:      {r1["model_used"]}')
print(f'Frontier model: {r2["model_used"]}')
if r1["model_used"] != r2["model_used"]:
    print('✅ Models are DIFFERENT')
else:
    print('⚠️ Same model — but through different code paths (X-Title differs)')
    print('   OSS:      X-Title: AI Assistant Evaluation (OSS)')
    print('   Frontier: X-Title: AI Assistant Evaluation (Frontier)')
print(f'{"=" * 60}')