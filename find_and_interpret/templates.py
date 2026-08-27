import re

# Plain-text sentinels prefixed to a user turn to tell the model which of
# its two functions to perform (seek a full page, judge a crop). Not new
# special tokens -- no embedding resize needed.
SEEK_MARKER = "[SEEK] "
JUDGE_MARKER = "[JUDGE] "

# Qwen3-VL's native grounding format: plain JSON {"bbox_2d": [x1,y1,x2,y2]},
# coordinates normalized to [0, 1000).
_QWEN_SCALE = 1000
_CALL_RE = re.compile(r'\{"bbox_2d":\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\}')
_MIN_SIDE = 8  # normalized (0.._QWEN_SCALE) units; thinner than this and the box is degenerate


def parse_seek_boxes(raw, w, h):
    """Every bbox_2d call in a [SEEK] response, scaled back to the page
    image's native (w, h) pixel space. Degenerate boxes are dropped
    individually rather than rejecting the whole response."""
    boxes = []
    for m in _CALL_RE.finditer(raw):
        gx1, gy1, gx2, gy2 = (int(g) for g in m.groups())
        if gx2 - gx1 < _MIN_SIDE or gy2 - gy1 < _MIN_SIDE:
            continue
        boxes.append((
            round(gx1 * w / _QWEN_SCALE),
            round(gy1 * h / _QWEN_SCALE),
            round(gx2 * w / _QWEN_SCALE),
            round(gy2 * h / _QWEN_SCALE),
        ))
    return boxes


def parse_judge_verdict(raw):
    """True/False/None (unparseable/truncated) -- checked in this order
    since "irrelevant." itself ends with "relevant."."""
    stripped = raw.strip()
    if stripped.endswith("irrelevant."):
        return False
    if stripped.endswith("relevant."):
        return True
    return None
