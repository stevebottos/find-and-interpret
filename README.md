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
from find_and_interpret.engine import DocVQAEngine, write_report
from find_and_interpret.utils import get_example_pages

QUESTIONS = [
    "What is the formula for computing scaled dot-product attention?",
    "Show the diagram of the overall Transformer model architecture.",
    "What is the formula used for positional encoding?",
    "Show the table comparing computational complexity, sequential operations, and maximum path length for self-attention, recurrent, and convolutional layers.",
    "What BLEU scores did the Transformer achieve on WMT 2014 English-to-German and English-to-French translation?",
    "Show the diagram of multi-head attention.",
    "What values of number of layers, model dimension, number of heads, and dropout were used for the base and big Transformer models?",
]

pages = get_example_pages()  # downloads + renders the sample paper, cached locally

engine = DocVQAEngine(pages, host="localhost", port=8000)
sections = []
for i, q in enumerate(QUESTIONS, 1):
    print(f"\n[{i}/{len(QUESTIONS)}] {q}")
    engine.set_question(q)
    boxes = engine.seek(n_workers=32)
    relevance = engine.judge(boxes, n_workers=32)
    sections.append(engine.visualize(boxes, relevance))

write_report(sections, "report.html")
```

`find_and_interpret.utils.get_example_pages()` fetches and renders the
"Attention Is All You Need" paper from arXiv on first call, caching the PDF
at `~/.cache/find_and_interpret/`.
