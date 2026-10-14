"""Step 2 (M2/M3) - cleaning + tokenisation.

Decisions (justify in report):
 * emojis are CONVERTED to words (":face_with_tears_of_joy:" -> "face with tears of joy"),
   because emojis are a strong sarcasm cue on Twitter;
 * URLs -> <url>, @mentions -> <user>;
 * hashtags are kept, with the '#' removed and CamelCase split (#NotAGoodDay -> not a good day);
 * we do NOT strip punctuation like '!' '?' '...' - they carry sarcasm signal; they become tokens;
 * stop-words are kept (words like "so", "totally", "just" matter for sarcasm).
"""
import re
import pandas as pd
from config import TAGGED_CSV, CLEAN_CSV

try:
    import emoji
    HAVE_EMOJI = True
except ImportError:
    HAVE_EMOJI = False

URL = re.compile(r"https?://\S+|www\.\S+")
USER = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
TOKEN = re.compile(r"<url>|<user>|[a-z]+(?:'[a-z]+)?|\d+|[!?]+|\.{2,}|[^\w\s]")


def split_camel(tag):
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
    return parts.replace("_", " ")


def convert_emoji(text):
    if HAVE_EMOJI:
        return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")
    return text  # fallback: emojis stay as symbols and are kept by the tokenizer


def clean_text(text, emoji_mode="convert"):
    t = str(text)
    t = URL.sub(" <url> ", t)
    t = USER.sub(" <user> ", t)
    t = HASHTAG.sub(lambda m: " " + split_camel(m.group(1)) + " ", t)
    if emoji_mode == "convert":
        t = convert_emoji(t)
    elif emoji_mode == "delete" and HAVE_EMOJI:
        t = emoji.replace_emoji(t, replace=" ")
    t = t.replace("&amp;", "&").replace("’", "'")
    t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def tokenize(clean):
    return TOKEN.findall(clean)


def main(emoji_mode="convert"):
    df = pd.read_csv(TAGGED_CSV)
    df["text_clean"] = df["text"].map(lambda x: clean_text(x, emoji_mode))
    df["tokens"] = df["text_clean"].map(lambda x: " ".join(tokenize(x)))
    empty = (df["tokens"].str.len() == 0).sum()
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved {len(df)} rows -> {CLEAN_CSV} (emoji_mode={emoji_mode}, "
          f"emoji package={'yes' if HAVE_EMOJI else 'NO - pip install emoji'}; empty after cleaning: {empty})")
    print(df[["text", "tokens"]].head(5).to_string(max_colwidth=70))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--emoji", choices=["convert", "delete", "keep"], default="convert")
    main(ap.parse_args().emoji)
