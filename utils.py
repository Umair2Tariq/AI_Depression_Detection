import re
import emoji

def clean_text(text):
    text = str(text).lower()

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Keep hashtag text, remove #
    text = re.sub(r"#", "", text)

    # Remove symbols except words
    text = re.sub(r"[^a-zA-Z0-9_\s:]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
