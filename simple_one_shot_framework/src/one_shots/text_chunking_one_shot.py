import tiktoken
from typing import Any, List
from chat_response import ChatResponse
from tool_use_one_shot import ToolUseOneShot


class TextChunkingOneShot(ToolUseOneShot):
    """TextChunkingOneShot is a subclass of ToolUseOneShot that processes large blocks of text in chunks"""

    def __init__(self,  **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.chunk_size = kwargs.get('chunk_size', 2000)
        self.encoder = tiktoken.encoding_for_model(self.model_name)

    def arun(self, input_data, overlap: int = 0):
        if isinstance(input_data, ChatResponse):
            input_data = input_data.text

        return await super().arun(self.chunk_paragraphs(input_data, overlap))

    def chunk_paragraphs(self, text: str, overlap: int = 0) -> List[str]:
        """
        Divide a block of text into chunks that fit within a token window using paragraphs as the quanta

        :param text: input string text to chunk
        :param overlap: number of paragraphs to overlap between chunks
        :returns: a list of chunked paragraphs

        """
        # Split text into paragraphs
        paragraphs = text.split("\n")

        # Array to hold result
        result = []

        # Temporary variable to hold paragraphs
        temp_paragraphs = []

        for paragraph in paragraphs:
            # Calculate tokens in the current paragraph
            tokens = len(self.encoder.encode(paragraph))

            # If tokens in the current paragraph and the temporary paragraphs exceed the limit
            # add the temporary paragraphs to the result and start a new temporary paragraphs
            if tokens + sum(len(self.encoder.encode(p)) for p in temp_paragraphs) > self.chunk_size:
                result.append('\n'.join(temp_paragraphs).strip())

                # Remove paragraphs from the start of temp_paragraphs to maintain overlap
                if overlap < len(temp_paragraphs):
                    temp_paragraphs = temp_paragraphs[-overlap:]

                temp_paragraphs.append(paragraph)
            else:
                # Otherwise, add the current paragraph to the temporary paragraphs
                temp_paragraphs.append(paragraph)

        # Don't forget to add the last batch of paragraphs
        if temp_paragraphs:
            result.append('\n'.join(temp_paragraphs).strip())

        return result
