from langchain_text_splitters import SentenceSplitter

def chunk_data(text):
    """
    Chunk the input text into smaller chunks, where each chunk is a sentence.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of chunks, where each chunk is a sentence.
    """
    splitter = SentenceSplitter()
    chunks = splitter.split(text)
    return chunks