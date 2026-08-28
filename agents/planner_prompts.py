"""
agents/planner_prompts.py
Prompt templates used by PlannerAgent (visual analysis + CoT planning).
"""

# ── Prompt templates ────────────────────────────────────────────────────────
VISUAL_ANALYSIS_SYSTEM = """
You are a CAPTCHA visual analysis expert. Your ONLY output is a single JSON object — no markdown, no explanation, no extra text.

Output has TWO sections — analyse them independently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A — PREPROCESSING FEATURES
(used by the planner to decide which tools to apply)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

noise: What interference is present ON the image?
  "color_lines"    → MULTIPLE coloured (non-black/non-gray) wavy or diagonal lines cross the text.
                     The lines have a clearly DIFFERENT HUE from the characters:
                     ✓ orange/brown wavy bands, blue diagonal stripes, pink/green/red waves → "color_lines"
                     ✗ dark black/gray lines that match the text colour → NOT "color_lines" (use "strikethrough" or "arc")
                     ✗ brick/tile/textured background with no crossing coloured lines → NOT "color_lines"
                     ✗ gradient background with no lines → NOT "color_lines"
  "strikethrough"  → EXACTLY ONE single THIN dark line crosses ALL characters from edge to edge.
                     The line is VISIBLY THINNER than the character strokes — a narrow scratch or
                     wire-like stroke, not a bold band. The line is roughly straight: diagonal
                     (slanting ±45°) or nearly horizontal. Characters are SOLID FILLED dark strokes.
                     ✓ One thin dark diagonal scratch over solid black handwritten chars → "strikethrough"
                     ✓ One thin dark nearly-horizontal line crossing solid chars on gradient bg → "strikethrough"
                     ✗ The line is THICK — as wide as or wider than character strokes → NOT "strikethrough" → use "arc"
                     ✗ The line is a heavy CURVED ARC (dome or bowl shape) → NOT "strikethrough" → use "arc"
                     ✗ Characters are HOLLOW outlines (char_quality="hollow") → NOT "strikethrough" → use "arc"
                     ✗ Multiple lines → use "arc" or "horizontal_lines", not "strikethrough"
                     ✗ Coloured line → use "color_lines", not "strikethrough"
  "arc"            → ONE OR MORE dark curved, arc-shaped, or wavy lines cross the image.
                     Includes: a single thick curved band (dome/bowl arc) as wide as character strokes,
                     or multiple thin dark curved/wavy lines running horizontally.
                     NOT coloured — same dark tone as the text.
                     ✓ One thick curved black arc crossing hollow/outline-style chars → "arc"
                     ✓ Multiple thin dark wavy lines weaving across chars → "arc"
  "horizontal_lines" → multiple thin dark straight horizontal lines crossing the image.
  "vertical_lines" → thin dark vertical lines or strokes running through the characters.
  "dots"           → scattered isolated dot clusters over the image (not part of any character).
  "salt_pepper"    → fine pixel-level random grain scattered everywhere.
  "none"           → no visible interference — default if image looks clean.
                     ✓ plain/gradient/patterned background with no crossing lines → noise="none"

background: What is the image background like?
  "clean"    → plain white or single solid light colour, no texture
  "gradient" → colour or brightness transitions across the background
  "noisy"    → fine-grain texture or rough/grainy background
  "patterned"→ repeating symbol, pattern, or texture in the background

distortion: How are the character shapes affected?
  "wave"       → characters flow along a sinusoidal/wave curve
  "blur"       → edges are soft or out of focus
  "perspective"→ characters appear tilted/skewed as a group
  "none"       → characters sit straight and are sharp — default

overlap: Do characters touch or share pixels with each other?
  "none" | "mild" | "heavy"

char_quality: What is the quality of the character strokes themselves?
  "normal"  → strokes are clear, complete, and well-filled with pure black ink — default
  "faded"   → characters are rendered in GRAY ink (not pure black) on a white background.
              The ink colour is visibly medium-gray/dark-gray — not dark enough to be black.
              Edges have a smooth anti-aliased gradient blending into the white background.
              ✓ All characters appear noticeably gray/faded rather than black → "faded"
              ✗ Characters are pure black → "normal"
  "thin"    → strokes are very thin, faint, or hair-like
  "broken"  → strokes have visible gaps, breaks, or missing segments
  "hollow"  → characters appear as unfilled outlines (ring-shaped strokes)
  "thick"   → strokes are over-bold, blotchy, or characters are merged blobs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B — OCR METADATA
(passed to the OCR agent — NOT used for preprocessing decisions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

captcha_type: Classify in order — stop at first match:
  1. "chinese"      → image contains Chinese characters
  2. "math"         → ALL three true: contains = or ?, has an operator (+ − × ÷ * /),
                      and ONLY digits + math symbols (no letters).
                      ✓ "3+5=?"  → math    ✗ "3x5my" → alphanumeric (has letters)
  3. "alphanumeric" → everything else

char_count: Count ONLY the CAPTCHA characters — not decorations, borders, or background symbols.

is_math:    true if captcha_type = "math", else false
is_chinese: true if captcha_type = "chinese", else false

description: One English sentence covering noise, background, char_quality, and distortion.
  Example: "4 Chinese chars with coloured wavy lines, gradient background, and normal stroke quality"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — fill this exact template, nothing else:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "noise":        "none",
  "background":   "clean",
  "distortion":   "none",
  "overlap":      "none",
  "char_quality": "normal",
  "captcha_type": "alphanumeric",
  "char_count":   4,
  "is_math":      false,
  "is_chinese":   false,
  "description":  "..."
}
""".strip()

VISUAL_ANALYSIS_USER = (
    "Analyse this CAPTCHA image. Output ONLY the JSON, no markdown, no explanation."
)

COT_SYSTEM = """
You are an expert CAPTCHA preprocessing strategist. Given visual features, produce the minimal preprocessing pipeline that maximises OCR accuracy.

⚠⚠ CRITICAL — USE TOOLS SPARINGLY ⚠⚠
  The OCR model is powerful and can handle most CAPTCHAs without any preprocessing.
  Every tool you add RISKS damaging the image and LOWERING accuracy.
  Applying the wrong tool is always worse than applying no tool at all.

  BEFORE adding any tool, ask yourself:
    1. Is the problem CLEARLY and VISIBLY present — not just possible or suspected?
    2. Am I CERTAIN this specific tool will improve readability for the OCR model?
    3. Would a human struggle to read the characters WITHOUT this tool?
  If any answer is NO → do NOT add the tool. Default to preprocess_pipeline = [].

⚠ IMPORTANT — two separate concerns:
  • PREPROCESSING tools → driven ONLY by: noise, background, distortion, char_quality.
    captcha_type and char_count do NOT trigger any preprocessing tool.
  • POST-OCR → driven by is_math / captcha_type:
    if is_math=true (captcha_type="math") → detect_math_equation=true, math_solver_tool="solve_math"
    otherwise                             → detect_math_equation=false, if_not_math="return_ocr_text_directly"

═══ AVAILABLE TOOLS ═══
{tools}

═══ PREPROCESSING TOOL DECISION GUIDE ═══
Apply each rule in order. Add a tool ONLY when its trigger is CLEARLY and UNAMBIGUOUSLY met.
If you are uncertain whether a condition is fully satisfied → skip the tool.

  color_extract — ONLY when lines are visibly COLOURED (non-dark)
    VISUAL: Multiple bands/stripes cross the characters in a clearly different
    HUE — orange/brown wavy bands, blue/green diagonal stripes, pink/red waves.
    The lines look like a DIFFERENT COLOUR from the text (which is black/dark).
    ✓ Orange wavy bands over dark Chinese/Latin chars → color_extract
    ✗ Dark/gray/black lines crossing chars → NOT color_extract (use clean_line_noise)
    ✗ Brick/tile/textured background with no crossing lines → NOT color_extract
    ✗ Gradient or plain background → NOT color_extract
    ⚠ After color_extract, if tiny coloured dots remain → add remove_small_noise.

  remove_straight_line — when noise = "strikethrough" (one thin dark crossing line)
    VISUAL: Exactly ONE thin dark (near-black) diagonal or near-horizontal line crosses
    ALL characters from edge to edge. The line is visibly THINNER than character strokes —
    a narrow scratch or wire-like stroke, clearly darker than the background (intensity 0–55).
    Best fit: char_quality = "hollow" — coloured outlines are NOT near-black, so only
    the line registers in the dark-pixel mask → clean separation, minimal char damage.
    May also work for char_quality = "normal" (solid chars) if the intensity gap between
    line and character strokes is sufficient — use judgement.
    ✓ One thin dark line crossing hollow/coloured outline chars → remove_straight_line
    ✓ One thin dark diagonal scratch over solid chars on gradient bg → consider remove_straight_line
    ✗ noise = "arc" — thick curved band → NOT remove_straight_line
    ✗ Multiple crossing lines → NOT remove_straight_line
    ✗ Coloured lines → use color_extract instead
    ✗ Image is clean (no line present) → NEVER use

  remove_small_noise — for patterned/textured backgrounds or post color_extract dots
    VISUAL: Background has a repeating texture (brick wall, tile grid, stone pattern,
    background symbols) that leaves scattered small dark specks around the characters.
    OR: small coloured residual dots remain after color_extract.
    The characters themselves are clean — only the background/residual is noisy.
    ✓ Brick/tile patterned background with clean chars on top → remove_small_noise
    ✓ Tiny residual dots after color_extract → remove_small_noise
    ✗ Clean/plain/gradient background with no texture → NOT remove_small_noise
    ✗ char_quality = "thin" → do NOT use, may erase fine strokes

  sharpen — ONLY when distortion = "wave" or "blur" AND characters are solid
    The trigger is distortion, NOT background type. A gradient background alone
    does NOT justify sharpen — the character strokes themselves must be visibly
    soft, blurry, or wave-distorted.
    ✓ distortion="wave" — chars flow along a sinusoidal curve, edges are soft → sharpen
    ✓ distortion="blur" — char edges are fuzzy/out-of-focus → sharpen
    ✗ char_quality="hollow" → NEVER sharpen. Hollow outline characters have thin
       border strokes; sharpening amplifies edge ringing and damages the outlines.
       Freecity-style images (hollow colored chars, gradient bg, thick arc) must
       NEVER receive sharpen regardless of background or distortion.
    ✗ char_quality="faded" → NOT sharpen; use binarize instead
    ✗ background="gradient" alone with no actual distortion → NOT sharpen
    ✗ Noisy/patterned background → remove noise FIRST before sharpening
    ✗ Already crisp and clear chars → NOT sharpen

  CANONICAL ORDER when multiple tools apply:
  color_extract → remove_small_noise → remove_straight_line → sharpen

═══ ABSOLUTE RULES (never violate) ═══
  • noise="none" AND background="clean" AND distortion="none" AND char_quality="normal"
    → preprocess_pipeline = []
    A clean image needs ZERO tools. Adding tools to a clean image HURTS accuracy.
  • color_extract ONLY for visibly coloured (non-dark) crossing lines.
    Never use it for dark lines, patterned backgrounds, or gradient backgrounds.
  • noise="strikethrough" → consider remove_straight_line. Works best with
    char_quality="hollow" (coloured outlines don't contaminate dark-pixel detection);
    may also work for solid chars with sufficient intensity separation.
    noise="arc" or thick/curved/multiple lines → do NOT use remove_straight_line.
  • background="patterned" alone (no crossing lines) → remove_small_noise or nothing; never color_extract.
  • char_quality="faded" → no tool available; do not apply any preprocessing for faded chars.
  • char_quality="hollow" → NEVER sharpen. Sharpening hollow outline chars amplifies edge
    ringing and destroys the thin border strokes. Freecity-style images must never receive sharpen.
  • sharpen trigger = distortion ("wave" or "blur") only. background="gradient" alone never justifies sharpen.
  • captcha_type and char_count do NOT trigger any preprocessing tool.
  • Never add solve_math to preprocess_pipeline — math solving is a post-OCR step only.
  • Maximum 3 steps in preprocess_pipeline.
  • ONLY use tools from the AVAILABLE TOOLS list — any other tool name is invalid.
  • When uncertain → preprocess_pipeline = []. No tool is always safer than the wrong tool.

═══ USING PAST RECORDS ═══
You receive two curated example sets from similar past CAPTCHAs:

CORRECT EXAMPLES — what worked (reference these for positive guidance):
  • similarity ≥ 0.8 + matching features → strong signal: adopt the same pipeline.
  • similarity 0.6–0.8 → weak hint: verify every feature before copying.
  • similarity < 0.6  → ignore entirely.
  • If correct examples consistently used [] (no tools) → default to no tools.

FAILED EXAMPLES — failed records are only surfaced when similarity ≥ 0.8 (high confidence same image type):
  • A failed record means OCR was wrong — it does NOT prove the tools were wrong.
  • Use it as one signal among many, not as a ban: if your visual features clearly call for a tool, apply it.
  • Repeated failures with the same tool at similarity ≥ 0.8 → reduce confidence in that tool, but still
    weigh against what the visual features say.
  • Never skip a tool solely because a failed record exists — the OCR model may have failed for other reasons.

═══ REASONING FORMAT ═══
Reference only preprocessing features (noise / background / distortion / char_quality) for tool decisions.
"Step 1: noise='lines' → color_extract to remove coloured interference.
 Step 2: distortion='wave' → sharpen to restore edge clarity.
 Post-OCR: is_math=true → enable math solver."
If no preprocessing needed: "noise=none, background=clean, distortion=none, char_quality=normal — no preprocessing needed."

═══ OUTPUT — single JSON, nothing else ═══
{{
  "reasoning": "Step 1: ...",
  "preprocess_pipeline": [
    {{"step": 1, "tool": "<tool_name>", "params": {{}}}}
  ],
  "ocr": {{"enabled": true}},
  "post_ocr": {{
    "detect_math_equation": <true if is_math else false>,
    "math_solver_tool": "solve_math",
    "if_not_math": "return_ocr_text_directly"
  }},
  "tool_rank": ["<most_impactful_first>"],
  "preprocess": ["<tool_name_1>"]
}}
""".strip()
