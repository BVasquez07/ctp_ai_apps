from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
It’s not just the specifications that are changing, either. Much has been made of permutations 
to Google’s algorithms, which are beginning to favor better written, more authoritative content (and making work for the growing content strategy industry). Google’s bots are now charged with asking questions like, 
“Was the article edited well, or does it appear sloppy or hastily produced?” and “Does this article provide a complete or comprehensive description of the topic?,” 
the sorts of questions one might expect to be posed by an earnest college professor.
"""

print(summarizer(text, max_length=130, min_length=30, do_sample=False))