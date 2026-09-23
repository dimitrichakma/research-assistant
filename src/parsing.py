import base64
import os

import fitz
from langchain_core.messages import HumanMessage
from langchain_typesafe import Noul, TypeSafeClassifier


def extract_columns(pdf_path):
    """Extract text from a PDF, preserving column reading order.

    Plain top-to-bottom reading of PyMuPDF's raw block order interleaves
    both columns of a two-column layout (a right-column block can have a
    lower y0 than a left-column block further down the page). Splitting
    blocks by which half of the page they start in, then sorting each
    half independently by vertical position, keeps each column coherent.
    """
    doc = fitz.open(pdf_path)
    full_text = []
    for page in doc:
        blocks = page.get_text("blocks")
        mid_x = page.rect.width / 2
        left = sorted([b for b in blocks if b[0] < mid_x], key=lambda b: b[1])
        right = sorted([b for b in blocks if b[0] >= mid_x], key=lambda b: b[1])
        for b in left + right:
            full_text.append(b[4])
    return "\n".join(full_text)


def extract_blocks(pdf_path):
    """Return every text block across the whole document, flattened.

    extract_columns() only returns joined text - Level 3's get_section()
    (src/tutor.py) needs the raw block structure instead, since its
    TypeSafe heading classification judges one block at a time. Ordering
    doesn't matter here the way it does in extract_columns(): get_section()
    only ever looks up a block's text as a substring of the already-built
    PAPER_TEXT (via .find()), so raw per-page PyMuPDF order is fine - no
    need to duplicate the left/right column sort for this.
    """
    doc = fitz.open(pdf_path)
    return [b for page in doc for b in page.get_text("blocks")]


def explain_visual(llm, image_path, kind="figure"):
    """Get a vision-based explanation of one cropped figure/equation image.

    llm is passed in, not constructed here - same rule as every other LLM
    call in this project (extract_paper(llm, text), etc.), so Level 4.5's
    BYOK swap and Level 3's tool wiring both work without a hardcoded
    model buried in this function. The roadmap's original sketch built a
    module-level `vision_llm = ChatAnthropic(...)` here, which actually
    violates that same rule - fixed here to match everything else.

    Claude reads the cropped image directly, no OCR - same mechanism for
    both figures and equations, just a different prompt.
    """
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()

    prompt = ("Describe this chart: type, axes, key trend." if kind == "figure"
              else "Explain this equation in plain English, term by term.")

    message = HumanMessage(content=[
        {"type": "image",
         "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": prompt}
    ])
    return llm.invoke([message]).content


def _safe_filename(text, max_len=20):
    """Turn arbitrary block text into a filesystem-safe filename fragment."""
    return "".join(c if c.isalnum() else "_" for c in text[:max_len]).strip("_")


def _merge_fragment_blocks(blocks, max_chars=60, max_gap=20):
    """Merge short, closely-stacked blocks into one region.

    PyMuPDF sometimes splits one equation into many tiny per-symbol
    blocks (operators, subscripts, summation bounds each as their own
    block, e.g. "max", "Phi", "X", "(x,y)eZ" as four separate blocks for
    one equation). Classifying each fragment alone almost never reads as
    "a standalone equation" on its own, so the equation gets silently
    missed entirely - confirmed empirically on LORA.pdf page 2, where a
    real numbered equation produced zero crops under per-fragment
    classification.

    Character count, not pixel width, is what actually separates
    equation fragments from real paragraphs - checked against a real
    page: every equation-related block was <=23 characters (even a
    visually "wide" line like "log (P_Phi(y_t|x,y_<t)) (1)", which is
    only 23 characters spread over 200+ points due to sparse symbol
    spacing), while every real paragraph block was 190+ characters. Pixel
    width was tried first and rejected - an equation's main horizontal
    line and a short paragraph both showed up around 200-260pt wide,
    with no reliable width cutoff separating them; character count had a
    huge, unambiguous margin instead.

    Grouping blocks that are both short and vertically close together
    into one combined region before classification fixes the miss.
    Normal paragraph text is left untouched, since a merge only triggers
    when BOTH blocks in a pair are short - a long paragraph immediately
    below a short header never merges with it.
    """
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda b: b[1])
    merged = [list(ordered[0])]
    for b in ordered[1:]:
        last = merged[-1]
        last_short = len(last[4].strip()) < max_chars
        this_short = len(b[4].strip()) < max_chars
        gap = b[1] - last[3]
        if last_short and this_short and gap < max_gap:
            merged[-1] = [
                min(last[0], b[0]), min(last[1], b[1]),
                max(last[2], b[2]), max(last[3], b[3]),
                last[4].rstrip() + " " + b[4].lstrip(),
                last[5], last[6],
            ]
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


def _crop_top_for_caption(caption, page_blocks, max_height=400, min_height=50,
                           max_label_chars=100):
    """Find how far above a caption its figure/table actually extends.

    A fixed guess (always N points above the caption) either clips
    figures taller than N or wastes space cropping blank page above
    figures shorter than N - confirmed on a real page: a 150pt fixed
    guess clipped the top off a tall architecture diagram.

    Stopping at the very first block found above the caption isn't right
    either - also confirmed on a real page: a multi-box architecture
    diagram has its own internal text labels ("User channels",
    "Orchestrator", ...) extracted as separate blocks scattered through
    the figure's vertical span, immediately above the caption. Stopping
    at the first one found crops almost nothing - just a sliver between
    the nearest label and the caption.

    Instead, walk upward through consecutive blocks that overlap the
    caption's horizontal range, extending the crop past each one that's
    short (a label embedded in the figure, not body prose), and only
    stopping at the first long block (real paragraph text) or once
    max_height is reached. max_label_chars separates the two: checked
    against a real page, this diagram's labels were 54-89 characters
    each, while the shortest real paragraph on the same page was 144 -
    a large, clean margin (a different, looser threshold than
    _merge_fragment_blocks' 60, since these are multi-word phrases, not
    single equation symbols).
    """
    cx0, cy0, cx1 = caption[0], caption[1], caption[2]
    above = sorted(
        (b for b in page_blocks
         if b is not caption and b[3] <= cy0 and b[0] < cx1 and b[2] > cx0),
        key=lambda b: -b[3],
    )
    top = cy0
    for b in above:
        if cy0 - b[1] > max_height or len(b[4].strip()) >= max_label_chars:
            break
        top = b[1]
    top = max(top, cy0 - max_height)
    top = min(top, cy0 - min_height)
    return max(0, top)


def extract_visuals(pdf_path, classifier: TypeSafeClassifier, output_dir="figures"):
    """Crop figures, table captions, and equations as standalone images.

    Returns a list of {"path": ..., "kind": "figure" | "equation"} dicts,
    not bare path strings - Level 4's supervisor_node (src/swarm.py) needs
    to know whether a paper has any equations at all (a free rule check,
    no LLM call) before deciding whether to run the Math node, which needs
    the kind tag, not just a file path.

    Classifies one page's blocks per TypeSafe call. Batching an entire
    document's blocks into a single call hits TypeSafe's per-request
    token limit on real papers - confirmed empirically: LORA.pdf's 628
    blocks (1256 questions) in one call raised max_tokens_exceeded, while
    a single page's ~20-30 blocks succeeds without issue.

    Each question's instructions name the specific block it's about
    (e.g. "Is the text in `block_3` a figure or table caption?") rather
    than a generic "Is this text a caption?" repeated for every block -
    without that, TypeSafe has no way to tell which block a given
    question refers to, since question IDs aren't sent to the model and
    every question in a batch sees the same shared state. Verified
    empirically: the generic phrasing produced near-coin-flip, backwards
    scores on two clearly-distinguishable blocks; naming the block
    directly fixed it.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    saved = []
    equation_count = {}  # page_num -> count, so multiple equations per page
    # don't overwrite each other's file
    caption_names = {}  # page_num -> set of filenames already used, so two
    # captions with the same first 20 sanitized characters don't collide

    for page_num, page in enumerate(doc):
        blocks = _merge_fragment_blocks(page.get_text("blocks"))
        if not blocks:
            continue

        questions = {}
        for i, _ in enumerate(blocks):
            questions[f"caption_{i}"] = Noul(
                instructions=f"Is the text in `block_{i}` a figure or table caption?")
            questions[f"equation_{i}"] = Noul(
                instructions=f"Is the text in `block_{i}` a standalone mathematical equation?")

        result = classifier.invoke({
            "state": {f"block_{i}": b[4] for i, b in enumerate(blocks)},
            "questions": questions,
        })

        for i, b in enumerate(blocks):
            text = b[4].strip()
            if result.nouls[f"caption_{i}"].noul > 0.5:
                top = _crop_top_for_caption(b, blocks)
                rect = fitz.Rect(b[0], top, b[2], b[1])
                pix = page.get_pixmap(clip=rect, dpi=200)
                used = caption_names.setdefault(page_num, set())
                base = _safe_filename(text) or "caption"
                name = base
                suffix = 1
                while name in used:
                    suffix += 1
                    name = f"{base}_{suffix}"
                used.add(name)
                path = f"{output_dir}/p{page_num}_{name}.png"
                pix.save(path)
                saved.append({"path": path, "kind": "figure"})
            elif result.nouls[f"equation_{i}"].noul > 0.5:
                rect = fitz.Rect(b[0], b[1], b[2], b[3])
                pix = page.get_pixmap(clip=rect, dpi=300)
                count = equation_count.get(page_num, 0)
                equation_count[page_num] = count + 1
                path = f"{output_dir}/p{page_num}_eq{count}.png"
                pix.save(path)
                saved.append({"path": path, "kind": "equation"})

    return saved


if __name__ == "__main__":
    text = extract_columns("docs/research_papers/paper 1.pdf")
    print(text[:1000])
