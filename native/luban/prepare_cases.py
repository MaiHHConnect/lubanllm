# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed text tokens without importing Torch or loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def prepare(model, fixtures):
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        (model / 'chat_template.jinja').read_text())
    def fail(message):
        raise ValueError(message)
    cases = []
    eos = json.loads((model / 'config.json').read_bytes())['text_config']['eos_token_id']
    for item in fixtures:
        prompt = template.render(messages=item['messages'], add_generation_prompt=True,
                                 enable_thinking=False, raise_exception=fail)
        tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
        cases.append({'id': item['id'], 'tokens': tokens, 'max_tokens': item['max_tokens'],
                      'eos_ids': [eos], 'dump_logits': item.get('dump_logits', False)})
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cases = prepare(args.model, json.loads(args.fixtures.read_bytes()))
    with args.output.open('x') as out:
        json.dump(cases, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({'cases': len(cases), 'tokens': [len(c['tokens']) for c in cases],
                      'cases_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
