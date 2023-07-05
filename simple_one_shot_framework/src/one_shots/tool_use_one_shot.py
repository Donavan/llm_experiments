import copy
import json
import openai
import asyncio
import inspect
import logging
from asyncio import Semaphore
from typing import Any, Dict, List, Union, Optional

from chat_response import ChatResponse
from simple_one_shot_framework.src.one_shots.chat_response import ChatResponse


class ToolUseOneShot:
    """
    A class used to represent a tool to execute tasks asynchronously.

    Attributes
    ----------
    model_name : str
        The name of the model to use, default is "gpt-3.5-turbo-16k"
    temperature : float
        The temperature parameter for the model, default is 0.5
    max_delay : int
        The maximum delay in seconds, default is 10
    concurrency_limit : int
        The maximum number of concurrent tasks, default is 3
    semaphore : Semaphore
        A semaphore instance to manage concurrency
    tool_chest : dict
        A dictionary of available tools
    prompt : str
        The prompt to use, default is None

    Methods
    -------
    __exponential_backoff(delay: int)
        Delays the execution for backoff strategy
    arun(input_data: Union[List, ChatResponse])
        Runs tasks asynchronously and returns the results
    """

    def __init__(self, **kwargs) -> None:
        """
        Constructs all the necessary attributes for the ToolUseOneShot object.

        Parameters
        ----------
        **kwargs :
             model_name : str
                The name of the model to use, default is "gpt-3.5-turbo-16k"
            temperature : float
                The temperature parameter for the model, default is 0.5
            max_delay : int
                The maximum delay in seconds, default is 10
            concurrency_limit : int
                The maximum number of concurrent tasks, default is 3
            semaphore : Semaphore
                A semaphore instance to manage concurrency
            tool_chest : dict
                A dictionary of available tools
            prompt : str
                The prompt to use, default is None
        """
        self.model_name: str = kwargs.get("model_name", "gpt-3.5-turbo-16k")
        self.temperature: float = kwargs.get("temperature", 0.5)
        self.max_delay: int = kwargs.get("max_delay", 10)
        self.concurrency_limit: int = kwargs.get("concurrency_limit", 3)
        self.semaphore: Semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.tool_chest: Dict[str, Any] = kwargs.get("tools", {})
        self.prompt: Optional[str] = kwargs.get("prompt", None)
        self.tool_chest['self'] = self
        self.schemas = None

    async def __exponential_backoff(self, delay: int) -> None:
        """
        Delays the execution for backoff strategy.

        Parameters
        ----------
        delay : int
            The delay in seconds.
        """
        await asyncio.sleep(min(2 * delay, self.max_delay))

    async def arun(self, input_data: Union[str, List[ChatResponse], ChatResponse, List[str]]) -> Union[List[ChatResponse], ChatResponse]:
        """
        Runs tasks asynchronously and returns the results.

        Parameters
        ----------
        input_data : Union[List, ChatResponse, str]
            The input data to process.
            Accepts a string, a ChatResponse, or an array of either

        Returns
        -------
        Union[List, dict]
            The results of the tasks.
        """
        if isinstance(input_data, list):
            tasks = []
            for input_text in input_data:
                if isinstance(input_text, ChatResponse):
                    input_text = input_text.text

                tasks.append(self._chat_one_shot(user_message=input_text))

            return await asyncio.gather(*tasks, return_exceptions=True)

        if isinstance(input_data, ChatResponse):
            input_data = input_data.text

        return await self._chat_one_shot(user_message=input_data)


    @staticmethod
    async def await_and_combine_multiple_results(tasks) -> ChatResponse:
        # Gather results and wait for completion while preserving order
        results = await asyncio.gather(*tasks, return_exceptions=True)

        text_chunks = []
        token_count = 0

        for index, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle the case where the task raised an exception
                print(f"An error occurred while processing a chat result: {str(result)}")
            else:
                text_chunks.append(result.text)
                token_count = token_count + result.token_count

        return ChatResponse('\n'.join(text_chunks), token_count)

    def __functions(self) -> List[Dict[str, Any]]:
        """
        Extracts JSON schemas from the objects in the toolchest

        :return: A list of JSON schemas.
        """
        if self.schemas is not None:
            return self.schemas

        self.schemas = []
        for src_name in self.tool_chest:
            for name, method in inspect.getmembers(self.tool_chest[src_name], predicate=inspect.ismethod):
                if hasattr(method, 'schema'):
                    schema = copy.deepcopy(method.schema)
                    schema['name'] = f"{src_name}-{schema['name']}"
                    self.schemas.append(schema)

        return self.schemas

    async def __call_function(self, function_id: str, function_args: dict):
        (toolbelt, function_name) = function_id.split("-", 2)
        src_obj = self.tool_chest[toolbelt]
        function_to_call = getattr(src_obj, function_name)

        return await function_to_call(**function_args)

    @staticmethod
    def __construct_message_array(additional_messages, user_message, sys_prompt):
        if user_message is None and additional_messages is None:
            raise ValueError("You must provide a user_message, message, or additional_messages")
        elif user_message is None:
            messages = [{"role": "system", "content": sys_prompt}] + additional_messages
        elif additional_messages is None:
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_message}]
        else:
            messages = [{"role": "system", "content": sys_prompt}] + additional_messages + [{"role": "user", "content": user_message}]

        return messages

    async def _chat_one_shot(self, **kwargs) -> ChatResponse:
        temperature: float = kwargs.get("temperature", self.temperature)
        model_name: str = kwargs.get("model_name", self.model_name)
        sys_prompt: str = kwargs.get("prompt", self.prompt)

        additional_messages: List[Dict[str, str]] = kwargs.get("additional_messages", None)
        user_message: str = kwargs.get("user_message", None)
        messages: List[Dict[str, str]] = kwargs.get("messages", None)

        if messages is None:
            if sys_prompt is None:
                raise ValueError("You must provide a user_message, message, or additional_messages")

            messages = self.__construct_message_array(additional_messages, user_message, sys_prompt)

        functions = self.__functions()
        opts = {"model": model_name, "temperature": temperature, "messages": messages}
        if len(functions):
            opts['functions'] = functions

        delay = 1  # Initial delay between retries
        async with self.semaphore:
            while True:
                try:
                    response = await openai.ChatCompletion.acreate(**opts)
                    response_message = response["choices"][0]["message"]
                    if response_message.get("function_call"):
                        function_response = await self.__call_function(response_message["function_call"]["name"], json.loads(response_message["function_call"]["arguments"]))
                        messages.append(response_message)
                        messages.append({"role": "function", "name": response_message["function_call"]["name"], "content": function_response})
                    else:
                        return ChatResponse(response_message["content"].strip(), response['usage']['completion_tokens'])

                except openai.error.InvalidRequestError as e:
                    logging.exception("Invalid request occurred: %s", e)
                    raise
                except openai.error.OpenAIError as e:
                    logging.exception("OpenAIError occurred: %s", e)
                    await self.__exponential_backoff(delay)
                    delay *= 2
                except Exception as e:
                    logging.exception("Error occurred during chat completion: %s", e)
                    raise
