# find-and-interpret

Document QA in two steps: find the region on a page that answers a question,
then interpret whether a candidate crop actually answers it. Internally this
is the model's two functions, [SEEK] and [JUDGE] which I renamed to "find" and "interpret" respectively.
Model: [`stevebottos/qwen3.5-0.8b-find-and-interpret`](https://huggingface.co/stevebottos/qwen3.5-0.8b-find-and-interpret),
served via vLLM for best results.

## Setup

```
make vllm-up          # starts vllm-seek on localhost:8000
uv sync
```

`make vllm-down` to stop it.

## Usage

```python
import json
import os

from PIL import Image

from find_and_interpret.engine import DocVQAEngine, write_report

DOC_DIR = "samples/attention_is_all_you_need"

pages = [
    (name, Image.open(f"{DOC_DIR}/pages/{name}").convert("RGB"))
    for name in sorted(os.listdir(f"{DOC_DIR}/pages"))
]
questions = json.load(open(f"{DOC_DIR}/questions.json"))

engine = DocVQAEngine(pages, host="localhost", port=8000)
sections = []
for i, q in enumerate(questions, 1):
    print(f"\n[{i}/{len(questions)}] {q}")
    engine.set_question(q)
    boxes = engine.seek(n_workers=32)
    relevance = engine.judge(boxes, n_workers=32)
    sections.append(engine.visualize(boxes, relevance))

write_report(sections, "report.html")
```

`samples/attention_is_all_you_need/` has a ready-to-use sample doc (page
images + questions) for trying this out.
