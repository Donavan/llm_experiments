import json
from typing import Any, Dict


class ChatResponse:
    """
    A class used to represent a Chat Response.

    Attributes
    ----------
    text : str
        The text response from the chat
    token_count : int
        The number of tokens in the response

    Methods
    -------
    json_object()
        Returns the JSON representation of the text response
    """

    def __init__(self, response_text: str, token_count: int) -> None:
        """
        Constructs all the necessary attributes for the Chat Response object.

        Parameters
        ----------
            response_text : str
                The text response from the chat
            token_count : int
                The number of tokens in the response
        """
        self.text = response_text
        self.token_count = token_count

    def json_object(self) -> Dict[str, Any]:
        """
        Returns the JSON representation of the text response.

        Returns
        -------
        dict
            The dictionary representation of the JSON text
        """
        return json.loads(self.text)
