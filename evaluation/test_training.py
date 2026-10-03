"""CPU regression for the actual final trainer's label masking."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from train_qwen_ray import encode_assistant_turns


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(text.encode('utf-8'))}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(text.encode('utf-8'))


def main():
    messages = [
        {'role': 'system', 'content': 'rules'},
        {'role': 'user', 'content': 'question'},
        {'role': 'assistant', 'content': 'answer one'},
        {'role': 'user', 'content': 'tool output'},
        {'role': 'assistant', 'content': 'answer two'},
    ]
    examples, stats = encode_assistant_turns([messages], FakeTokenizer(), 512)
    assert len(examples) == 2 and stats['assistant_turns'] == 2
    for example in examples:
        first = next(i for i, value in enumerate(example['labels']) if value != -100)
        assert first > 0
        assert all(value == -100 for value in example['labels'][:first])
        assert all(label == token for label, token in zip(example['labels'], example['input_ids']) if label != -100)
    messages[0]['content'] = 'x' * 400
    examples, stats = encode_assistant_turns([messages[:3]], FakeTokenizer(), 96)
    assert len(examples) == 1 and stats['left_truncated'] == 1
    assert any(value != -100 for value in examples[0]['labels'])
    print('PASS: final-trainer assistant masking, multiple turns, target-preserving truncation')


if __name__ == '__main__':
    main()
