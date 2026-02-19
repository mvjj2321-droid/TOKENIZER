"""
Tokenizer Explorer — Python Backend
Uses the real tiktoken library for 100% accurate GPT-2 / GPT-4 token IDs.
Falls back to embedded vocab if tiktoken files are not cached yet.

SETUP (run once):
    pip install tiktoken transformers
    python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('gpt2')"
    python3 -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')"
"""

from flask import Flask, request, jsonify, send_from_directory
import re

app = Flask(__name__, static_folder="static")

# ══════════════════════════════════════════════════════════════
#  LOAD REAL TOKENIZERS via tiktoken + transformers
# ══════════════════════════════════════════════════════════════

TIKTOKEN_AVAILABLE = False
BERT_TOKENIZER = None

try:
    import tiktoken
    ENC_GPT2 = tiktoken.get_encoding("gpt2")
    ENC_GPT4 = tiktoken.get_encoding("cl100k_base")
    TIKTOKEN_AVAILABLE = True
    print("✓ tiktoken loaded — using real GPT-2 and GPT-4 token IDs")
except Exception as e:
    print(f"✗ tiktoken not available ({e}) — using fallback vocab")

try:
    from transformers import BertTokenizer
    BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    print("✓ BERT tokenizer loaded — using real BERT token IDs")
except Exception as e:
    print(f"✗ BERT tokenizer not available ({e}) — using fallback vocab")


# ══════════════════════════════════════════════════════════════
#  WORD LIST for word-level model
# ══════════════════════════════════════════════════════════════

WORD_LIST = [
    "the","of","and","a","to","in","is","you","that","it","he","was","for",
    "on","are","as","with","his","they","at","be","this","have","from","or",
    "one","had","by","but","not","what","all","were","we","when","your","can",
    "said","there","an","which","she","do","how","their","if","will","up",
    "other","about","out","many","then","them","these","so","some","her",
    "would","make","like","him","into","time","has","look","two","more","go",
    "see","no","way","could","people","my","than","first","been","call","who",
    "its","now","find","long","down","day","did","get","come","may","over",
    "new","only","work","know","place","year","me","back","give","most","very",
    "after","our","just","name","good","man","think","say","great","where",
    "help","much","before","right","too","old","same","tell","want","show",
    "also","three","small","set","end","need","big","high","here","why","ask",
    "read","hand","always","move","city","still","learn","number","hope","love",
    "life","happy","sad","angry","fear","beautiful","fast","slow","hot","cold",
    "light","dark","start","stop","open","close","yes","please","thank","sorry",
    "home","school","road","car","tree","water","fire","earth","sky","sun",
    "moon","star","dream","heart","mind","soul","power","truth","future","past",
    "memory","story","book","word","music","art","science","red","blue","green",
    "black","white","color","voice","image","video","game","win","lose","try",
    "fail","build","create","change","grow","run","walk","jump","fall","rise",
    "begin","feel","hear","touch","Jason","John","Mary","David","James","token",
    "model","text","data","language","neural","network","deep","machine",
    "learning","intelligence","computer","code","program","function","variable",
    "array","string","boolean","object","class","method","return","import",
    "const","let","var","true","false","null","algorithm","training","input",
    "output","error","success","hello","world","Python","Flask","API","server",
    "dashboard","Ankita","tokenizer","embedding","attention","transformer",
]

WORD_T2I = {w.lower(): i for i, w in enumerate(WORD_LIST)}
WORD_I2T = {i: w for i, w in enumerate(WORD_LIST)}


# ══════════════════════════════════════════════════════════════
#  TOKENIZE  (real tiktoken when available)
# ══════════════════════════════════════════════════════════════

def tokenize_with_tiktoken(text: str, enc) -> list[dict]:
    """
    Use tiktoken to tokenize — returns real token IDs and decoded text.
    Each token ID is decoded back to its string using enc.decode_single_token_bytes().
    """
    ids = enc.encode(text)
    result = []
    for token_id in ids:
        # Decode each token individually to get its text
        try:
            token_bytes = enc.decode_single_token_bytes(token_id)
            token_text = token_bytes.decode("utf-8", errors="replace")
        except Exception:
            token_text = f"<{token_id}>"
        result.append({
            "text": token_text,
            "id": token_id,
            "original": token_text.strip()
        })
    return result


def tokenize_bert_real(text: str) -> list[dict]:
    """Use HuggingFace BERT tokenizer for real WordPiece token IDs."""
    encoding = BERT_TOKENIZER(text, add_special_tokens=False)
    ids = encoding["input_ids"]
    tokens = BERT_TOKENIZER.convert_ids_to_tokens(ids)
    return [
        {"text": tok, "id": tid, "original": tok.replace("##", "")}
        for tok, tid in zip(tokens, ids)
    ]


def tokenize_char(text: str) -> list[dict]:
    result = []
    for ch in text:
        display = "·" if ch == " " else ("↵" if ch == "\n" else ch)
        result.append({"text": display, "id": ord(ch), "original": ch})
    return result


def tokenize_word(text: str) -> list[dict]:
    if not text.strip():
        return []
    result = []
    for tok in re.findall(r"\S+|\s+", text):
        word = tok.strip().lower()
        token_id = WORD_T2I.get(word)
        result.append({"text": tok, "id": token_id, "original": tok})
    return result


def run_tokenizer(text: str, model: str) -> list[dict]:
    if model == "char":
        return tokenize_char(text)
    if model == "word":
        return tokenize_word(text)
    if model == "bert":
        if BERT_TOKENIZER:
            return tokenize_bert_real(text)
        return [{"text": text, "id": None, "original": text, "error": "BERT not loaded"}]
    if model in ("gpt2", "gpt4"):
        if TIKTOKEN_AVAILABLE:
            enc = ENC_GPT2 if model == "gpt2" else ENC_GPT4
            return tokenize_with_tiktoken(text, enc)
        return [{"text": text, "id": None, "original": text, "error": "tiktoken not loaded"}]
    return []


# ══════════════════════════════════════════════════════════════
#  DECODE  (token ID → text)
# ══════════════════════════════════════════════════════════════

def decode_token(token_id: int, model: str) -> dict:
    if model == "char":
        try:
            ch = chr(token_id)
            display = "·" if ch == " " else ("↵" if ch == "\n" else ch)
            return {"id": token_id, "text": display, "found": True}
        except Exception:
            return {"id": token_id, "text": None, "found": False}

    if model == "word":
        text = WORD_I2T.get(token_id)
        return {"id": token_id, "text": text, "found": text is not None}

    if model in ("gpt2", "gpt4") and TIKTOKEN_AVAILABLE:
        enc = ENC_GPT2 if model == "gpt2" else ENC_GPT4
        try:
            token_bytes = enc.decode_single_token_bytes(token_id)
            text = token_bytes.decode("utf-8", errors="replace")
            return {"id": token_id, "text": text, "found": True}
        except Exception:
            return {"id": token_id, "text": None, "found": False}

    if model == "bert" and BERT_TOKENIZER:
        try:
            text = BERT_TOKENIZER.convert_ids_to_tokens([token_id])[0]
            found = text != "[UNK]"
            return {"id": token_id, "text": text, "found": found}
        except Exception:
            return {"id": token_id, "text": None, "found": False}

    return {"id": token_id, "text": None, "found": False}


# ══════════════════════════════════════════════════════════════
#  QUICK PICKS
# ══════════════════════════════════════════════════════════════

QUICK_PICK_WORDS = {
    "gpt2": ["the","and","hope","love","world","life","Jason","dashboard","Ankita","const","token","learning","hello","data","model"],
    "gpt4": ["the","and","hope","love","world","life","Jason","dashboard","Ankita","const","token","learning","hello","data","model"],
    "bert": ["the","and","hope","love","world","life","jason","dashboard","ankita","const","token","learning","hello","data","model"],
    "char": ["H","O","P","E"," ","d","a","s","h","b","o","r"],
    "word": ["hello","world","hope","love","life","the","and","dashboard","Ankita","token","model","data"],
}

def get_quick_picks(model: str) -> list[dict]:
    picks = []
    for word in QUICK_PICK_WORDS.get(model, []):
        tokens = run_tokenizer(word, model)
        if tokens and tokens[0].get("id") is not None:
            picks.append({"word": word, "id": tokens[0]["id"]})
    return picks


# ══════════════════════════════════════════════════════════════
#  STATUS ENDPOINT — tells frontend which tokenizers are loaded
# ══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({
        "tiktoken": TIKTOKEN_AVAILABLE,
        "bert": BERT_TOKENIZER is not None,
        "models": {
            "gpt2": TIKTOKEN_AVAILABLE,
            "gpt4": TIKTOKEN_AVAILABLE,
            "bert": BERT_TOKENIZER is not None,
            "char": True,
            "word": True,
        }
    })


# ══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/tokenize", methods=["POST"])
def api_tokenize():
    body = request.get_json(force=True)
    text = body.get("text", "")
    model = body.get("model", "gpt4")
    tokens = run_tokenizer(text, model)
    word_count = len(text.split()) if text.strip() else 0
    return jsonify({
        "tokens": tokens,
        "count": len(tokens),
        "char_count": len(text),
        "word_count": word_count,
        "chars_per_token": round(len(text) / len(tokens), 2) if tokens else None,
        "engine": "tiktoken" if TIKTOKEN_AVAILABLE and model in ("gpt2","gpt4") else
                  "bert-tokenizer" if BERT_TOKENIZER and model == "bert" else "builtin",
    })


@app.route("/api/decode", methods=["POST"])
def api_decode():
    body = request.get_json(force=True)
    ids = body.get("ids", [])
    model = body.get("model", "gpt4")
    decoded = [decode_token(int(tid), model) for tid in ids]
    return jsonify({"tokens": decoded})


@app.route("/api/quickpicks/<model>")
def api_quickpicks(model):
    return jsonify(get_quick_picks(model))


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Tokenizer Explorer")
    print("="*50)
    print(f"tiktoken (GPT-2/GPT-4): {'✓ REAL' if TIKTOKEN_AVAILABLE else '✗ fallback'}")
    print(f"BERT tokenizer:         {'✓ REAL' if BERT_TOKENIZER else '✗ fallback'}")
    print("="*50)
    if not TIKTOKEN_AVAILABLE:
        print("\nTo get 100% accurate results, run:")
        print("  python3 -c \"import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('gpt2')\"")
    print()
    app.run(debug=True, port=5000)
