from haystack.components.generators import OpenAIGenerator
from haystack import component, logging
from typing import List, Dict, Any, Optional, Callable, Union
from haystack.dataclasses import StreamingChunk
from haystack.dataclasses import ChatMessage
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from openai import Stream

logger = logging.getLogger(__name__)

class CustomOpenAIGenerator(OpenAIGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _connect_chunks(chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.
        """
        complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        complete_response.meta.update(
            {
                "model": chunk.model,
                "index": 0,
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": {},  # we don't have usage data for streaming responses
            }
        )
        return complete_response

    @staticmethod
    def _build_message(completion: Any, choice: Any) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.

        :param completion:
            The completion returned by the OpenAI API.
        :param choice:
            The choice returned by the OpenAI API.
        :returns:
            The ChatMessage.
        """
        # function or tools calls are not going to happen in non-chat generation
        # as users can not send ChatMessage with function or tools calls
        chat_message = ChatMessage.from_assistant(choice.message.content or "")
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage),
            }
        )
        return chat_message

    @staticmethod
    def _check_finish_reason(message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.

        If the `finish_reason` is `length`, log a warning to the user.

        :param message:
            The message returned by the LLM.
        """
        if message.meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
        if message.meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index {index} has been truncated due to the content filter.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt:
            The string prompt to use for text generation.
        :param system_prompt:
            The system prompt to use for text generation. If this run time system prompt is omitted, the system
            prompt, if defined at initialisation time, is used.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. For more details on the parameters supported by the OpenAI API, refer to
            the OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :returns:
            A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        message = ChatMessage.from_user(prompt)
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt), message]
        elif self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self.streaming_callback
        def convert_message_to_openai_format(message: ChatMessage) -> Dict[str, Any]:
            openai_msg = {"role": message.role.value, "content": message.text}
            if message.name:
                openai_msg["name"] = message.name

            return openai_msg
        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [convert_message_to_openai_format(message) for message in messages]

        completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore
            stream=streaming_callback is not None,
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        if isinstance(completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            chunk = None

            # pylint: disable=not-an-iterable
            for chunk in completion:
                if chunk.choices and streaming_callback:
                    streaming_callback(chunk)  # invoke callback with the chunk_delta
            completions = [self._connect_chunks(chunk, chunks)]
        elif isinstance(completion, ChatCompletion):
            completions = [self._build_message(completion, choice) for choice in completion.choices]

        # before returning, do post-processing of the completions
        for response in completions:
            self._check_finish_reason(response)

        return {
            "replies": [message.text for message in completions],
            "meta": [message.meta for message in completions],
        }