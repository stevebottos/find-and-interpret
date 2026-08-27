"""DocVQAEngine: seek+judge inference against a vLLM-served checkpoint.
One engine per document -- holds its pages (resized to seek's training
scale internally), reused across questions:

    engine = DocVQAEngine(pages, host="localhost", port=8000)
    engine.set_question("blah blah blah")
    boxes = engine.seek(n_workers=24)
    relevance = engine.judge(boxes, n_workers=24)
    html = engine.visualize(boxes, relevance)
    hits = engine.relevant_crops(boxes, relevance)

seek()/judge() each run concurrently over their pages/crops via aiohttp,
capped at n_workers in-flight requests -- vLLM's own scheduler does the
actual batching, not this code.
"""

import asyncio
import base64
import io
import itertools

import aiohttp
from PIL import Image, ImageDraw

from find_and_interpret.templates import (
    JUDGE_MARKER,
    SEEK_MARKER,
    parse_judge_verdict,
    parse_seek_boxes,
)

STYLE = """
body{font-family:system-ui;background:#111;color:#eee;padding:20px}
img{border:1px solid #444;margin:4px 0}
.q{border-top:3px solid #666;padding:20px 0}
.hit{border-top:1px solid #333;padding:10px 0;display:flex;flex-direction:column;align-items:center}
.page img{max-width:1100px;width:100%;height:auto}
.crops{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.crops div{text-align:center}
.ok{color:#4c9}.bad{color:#d55}
pre{background:#1a1a1a;padding:8px;border-radius:4px;white-space:pre-wrap;text-align:left;max-width:1100px}
.summary{border-bottom:3px solid #666;padding-bottom:20px;margin-bottom:20px}
.summary-q{border-top:1px solid #333;padding:10px 0}
.summary-q h3{margin:0 0 8px}
.summary-crops{display:flex;flex-direction:column;gap:8px}
.summary-item{display:flex;flex-direction:row;align-items:flex-start;gap:12px}
.summary-item img{flex:0 0 auto;max-width:220px;max-height:220px;width:auto;height:auto}
.summary-item .text{flex:1;min-width:0;text-align:left}
.summary-item .meta{color:#999;margin-bottom:4px}
.summary-item pre{max-width:none}
"""


def _b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _img_tag(img):
    return f"<img src='data:image/png;base64,{_b64(img)}'>"


def _with_boxes(img, boxes):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, outline="#3af", width=3)
    return img


def _resize_to_train_scale(img):
    scale = min(1.0, 800 / max(img.size))  # seek's training-scale page size
    return img.resize(tuple(round(x * scale) for x in img.size), Image.LANCZOS) if scale < 1 else img


def _summary_html(relevant_crops):
    """One <div class='summary'> block listing every relevant crop found,
    grouped by question, each with its source page and judge justification."""
    groups_html = []
    for question, crops in itertools.groupby(relevant_crops, key=lambda c: c["question"]):
        crops = list(crops)
        crops_html = "".join(
            f"<div class='summary-item'>{_img_tag(c['img'])}"
            f"<div class='text'><div class='meta'>from {c['page']} &mdash; "
            f"<span class='{'ok' if c['relevant'] else 'bad'}'>relevant={c['relevant']}</span></div>"
            f"<pre>{c['raw']}</pre></div></div>"
            for c in crops
        )
        groups_html.append(
            f"<div class='summary-q'><h3>{question} ({len(crops)} relevant crop{'s' if len(crops) != 1 else ''})</h3>"
            f"<div class='summary-crops'>{crops_html}</div></div>"
        )
    return f"<div class='summary'><h2>Relevant crops ({len(relevant_crops)})</h2>{''.join(groups_html)}</div>"


def write_report(sections, out_path, relevant_crops=None):
    """Join per-question visualize() sections into one standalone HTML file,
    with an optional summary of every relevant crop at the top."""
    summary = _summary_html(relevant_crops) if relevant_crops is not None else ""
    with open(out_path, "w") as f:
        f.write(
            f"<html><head><meta charset='utf-8'><style>{STYLE}</style></head><body>\n"
            + summary
            + "\n".join(sections)
            + "\n</body></html>\n"
        )


class DocVQAEngine:
    def __init__(
        self,
        pages,
        host="localhost",
        port=8000,
        endpoint="/v1/chat/completions",
        model="seek",
        seek_max_tokens=200,
        judge_max_tokens=300,
    ):
        # judge_max_tokens > seek_max_tokens: judge's own reasoning trace
        # sometimes needs >200 (seen truncating mid-verdict)
        self.pages = [(name, _resize_to_train_scale(img)) for name, img in pages]
        self.url = f"http://{host}:{port}{endpoint}"
        self.model = model
        self.seek_max_tokens = seek_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.question = None

    def set_question(self, question):
        self.question = question

    async def _generate(self, session, sem, query, img, max_tokens):
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_b64(img)}"},
                        },
                    ],
                }
            ],
        }
        async with sem, session.post(self.url, json=body) as resp:
            data = await resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _seek_page(self, session, sem, name, img):
        w, h = img.size
        raw = await self._generate(
            session, sem, SEEK_MARKER + self.question, img, self.seek_max_tokens
        )
        boxes = parse_seek_boxes(raw, w, h)
        print(f"[seek] {name}: {len(boxes)} region(s) found")
        return {"page": name, "img": img, "raw": raw, "boxes": boxes}

    def seek(self, n_workers=24):
        """Runs [SEEK] concurrently over every page. Returns a list of
        per-page dicts (page, img, raw, boxes) -- boxes is [] for pages
        deemed not relevant."""
        assert self.question, "call set_question() first"

        async def _run():
            sem = asyncio.Semaphore(n_workers)
            async with aiohttp.ClientSession() as session:
                return await asyncio.gather(
                    *(
                        self._seek_page(session, sem, name, img)
                        for name, img in self.pages
                    )
                )

        return asyncio.run(_run())

    async def _judge_page(self, session, sem, page):
        async def _one(i, box):
            crop = page["img"].crop(tuple(round(c) for c in box))
            raw = await self._generate(
                session, sem, JUDGE_MARKER + self.question, crop, self.judge_max_tokens
            )
            relevant = parse_judge_verdict(raw)
            verdict = "relevant" if relevant else "irrelevant" if relevant is False else "unparsed"
            print(f"[judge] {page['page']} region {i}: {verdict}")
            return {"box": box, "raw": raw, "relevant": relevant}

        return await asyncio.gather(*(_one(i, box) for i, box in enumerate(page["boxes"])))

    def judge(self, boxes, n_workers=24):
        """Runs [JUDGE] concurrently over every predicted crop from seek()'s
        output. Returns {page_name: [{box, raw, relevant}, ...]}, only for
        pages that had at least one predicted box."""
        pages_with_boxes = [p for p in boxes if p["boxes"]]

        async def _run():
            sem = asyncio.Semaphore(n_workers)
            async with aiohttp.ClientSession() as session:
                crops = await asyncio.gather(
                    *(self._judge_page(session, sem, p) for p in pages_with_boxes)
                )
            return dict(zip((p["page"] for p in pages_with_boxes), crops))

        return asyncio.run(_run())

    def relevant_crops(self, boxes, relevance):
        """Returns [{question, page, img, raw}, ...] for every crop judge
        marked relevant for the current question -- feed these into
        write_report(..., relevant_crops=...) for the top-of-report summary."""
        return [
            {
                "question": self.question,
                "page": page["page"],
                "img": page["img"].crop(tuple(round(c) for c in ce["box"])),
                "raw": ce["raw"],
                "relevant": ce["relevant"],
            }
            for page in boxes
            for ce in relevance.get(page["page"], [])
            if ce["relevant"]
        ]

    def visualize(self, boxes, relevance):
        """Returns one <div class='q'> HTML section for the current question."""
        hits_html = []
        for page in boxes:
            crops = relevance.get(page["page"], [])
            if not crops:
                continue
            crops_html = "".join(
                f"<div>{_img_tag(page['img'].crop(tuple(round(c) for c in ce['box'])))}"
                f"<br><span class='{'ok' if ce['relevant'] else 'bad'}'>relevant={ce['relevant']}</span>"
                f"<pre>{ce['raw']}</pre></div>"
                for ce in crops
            )
            hits_html.append(
                f"<div class='hit'><h4>{page['page']}</h4>"
                + f"<div class='page'>{_img_tag(_with_boxes(page['img'], [ce['box'] for ce in crops]))}</div>"
                + f"<pre>{page['raw']}</pre><div class='crops'>{crops_html}</div></div>"
            )
        return (
            f"<div class='q'><h2>{self.question}</h2>"
            + ("".join(hits_html) if hits_html else "<p>no page deemed relevant</p>")
            + "</div>"
        )
