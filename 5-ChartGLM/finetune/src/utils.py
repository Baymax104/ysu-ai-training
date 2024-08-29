# -*- coding: UTF-8 -*-
import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Union

from ruamel import yaml


def resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser
