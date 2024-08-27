import openai
import requests
import os 
import enum
from unify.clients import Unify,AsyncUnify # type: ignore
from forge.json.parsing import json_loads
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge import ChatProvider 
from .._openai_base import BaseOpenAIChatProvider, BaseOpenAIEmbeddingProvider
from ..schema import (
    ChatModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    EmbeddingModelInfo, 
)
from ..openai import *

class UnifyAiModelName(str, enum.Enum):
    mixtral_8x7b_instruct_v1 = "mixtral-8x7b-instruct-v0.1"
    mistral_7b_instruct_v2 = "mistral-7b-instruct-v0.2"
    mixtral_8x22b_instruct_v1 = "mixtral-8x22b-instruct-v0.1"
    qwen_2_72b_instruct = "qwen-2-72b-instruct"
    codellama_7b_instruct = "codellama-7b-instruct"
    codellama_70b_instruct = "codellama-70b-instruct"
    gpt_3_turbo = "gpt-3.5-turbo"
    gpt_4_turbo = "gpt-4-turbo"
    claude_3_opus = "claude-3-opus"
    llama_3_8b_chat = "llama-3-8b-chat"
    codellama_34b_instruct = "codellama-34b-instruct"
    mistral_7b_instruct_v3 = "mistral-7b-instruct-v0.3"
    mistral_7b_instruct_v01 = "mistral-7b-instruct-v0.1"
    gemma_2_9b_it = "gemma-2-9b-it"
    phind_codellama_34b_v2 = "phind-codellama-34b-v2"
    phind_codellama_34b_v1 = "phind-codellama-34b-v1"
    qwen_2_7b_instruct = "qwen-2-7b-instruct"
    deepseek_coder_33b_instruct = "deepseek-coder-33b-instruct"
    mistral_small = "mistral-small"
    claude_3_sonnet = "claude-3-sonnet"
    llama_3_70b_chat = "llama-3-70b-chat"
    gemma_7b_it = "gemma-7b-it"
    llama_3_1_8b_chat = "llama-3.1-8b-chat"
    codellama_13b_instruct = "codellama-13b-instruct"
    llama_31_405b_chat = "llama-3.1-405b-chat"
    mistral_large = "mistral-large"
    gpt_4 = "gpt-4"
    gemma_2b_it= "gemma-2b-it"
    claude_3_haiku = "claude-3-haiku"
    gpt_4o = "gpt-4o"
    llama_3_8b_chat@together-ai = "llama-3-8b-chat@together-ai"
    llama_3_70b_chat = "llama-3.1-70b-chat"
    llama_3_8b_chat = "llama-3-8b-chat"

UNIFY_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name=UnifyAiModelName.mixtral_8x7b_instruct_v1,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0001 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.mixtral_8x22b_instruct_v1,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00002 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.gemma_2_9b_it,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.deepseek_coder_33b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.mistral_large,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.mistral_small,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.gemma_7b_it,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
        EmbeddingModelInfo(
            name=UnifyAiModelName.gemma_2b_it,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
    ]
}


UnifyChatModels = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=UnifyAiModelName.qwen_2_72b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.codellama_7b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.codellama_70b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.gpt_3_turbo,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.gpt_4_turbo,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.claude_3_opus,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_8b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.codellama_34b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.mistral_7b_instruct_v3,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.mistral_7b_instruct_v01,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.mistral_7b_instruct_v2,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.phind_codellama_34b_v2,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.phind_codellama_34b_v1,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.qwen_2_7b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.claude_3_sonnet,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_70b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.codellama_13b_instruct,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_31_405b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_1_8b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.gpt_4,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.claude_3_haiku,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.gpt_4o,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_8b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_70b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=UnifyAiModelName.llama_3_8b_chat,
            provider_name=ModelProviderName.UNIFYAI,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=32768,
            has_function_call_api=False,
        ),
    ]
}

class UnifyAIError(Exception):
    """Base exception for UnifyAI-related errors."""
    def __init__(self, message: str, code: Optional[str] = None):
        super()._init_(message)
        self.code = code
        
class UnifyAIRateLimitError(UnifyAIError):
    """Exception for rate limit errors."""
    def __init__(self, message="Rate limit exceeded.", retry_after=None, limit=None):
        """
        Initialize the rate limit error with additional details.

        Args:
            message (str): Error message.
            retry_after (int): Time in seconds after which the request can be retried.
            limit (int): The number of requests allowed within the rate limit window.
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit

    def __str__(self):
        """
        Return a string representation of the error including retry information.
        """
        base_message = super().__str__()
        if self.retry_after:
            return f"{base_message} Please retry after {self.retry_after} seconds."
        return base_message

class UnifyAICredentials(ModelProviderCredentials):
    """Credentials for UnifyAI."""

    unifyai_api_key: SecretStr = UserConfigurable(from_env="UNIFYAI_API_KEY")
    unifyai_base_url: Optional[SecretStr] = UserConfigurable(
        default="https://api.unify.ai", from_env="UNIFYAI_BASE_URL"
    )

    def get_api_access_kwargs(self) -> Dict[str, Any]:
        return {
            "unifyai_api_key": self.unifyai_api_key.get_secret_value(),
            "unifyai_base_url": self.unifyai_base_url.get_secret_value(),
        }

    def deploy_model(self, model_name: str, model_file_path: str) -> Dict[str, Any]:
        """Example method to deploy a model to UnifyAI."""
        api_access_kwargs = self.get_api_access_kwargs()
        # Implement API call to UnifyAI to deploy the model
        response = self._call_unifyai_api(
            "POST", f"{api_access_kwargs['unifyai_base_url']}/models/deploy",
            headers={"Authorization": f"Bearer {api_access_kwargs['unifyai_api_key']}"},
            json={"model_name": model_name, "model_file_path": model_file_path},
        )
        return response.json()

    def _call_unifyai_api(self, method: str, url: str, headers: Dict[str, str], json: Dict[str, Any]) -> Any:
        import requests
        response = requests.request(method, url, headers=headers, json=json)
        response.raise_for_status()
        return response

<<<<<<< HEAD
class UnifyAIProvider(BaseOpenAIChatProvider[UnifyAIModelName, UnifyAISettings],
    BaseOpenAIEmbeddingProvider[UnifyAIModelName, UnifyAISettings],):
=======
class UnifyAISettings(ModelProviderSettings):
    credentials: Optional[UnifyAICredentials] = None
    budget: ModelProviderBudget = ModelProviderBudget()
    rate_limit_requests: int = Field(default=60, description="Number of requests allowed per minute")
    rate_limit_tokens: int = Field(default=250000, description="Number of tokens allowed per minute")
    
class UnifyAIProvider(BaseOpenAIChatProvider[UnifyAiModelName, UnifyAISettings],
                      BaseOpenAIEmbeddingProvider[UnifyAiModelName, UnifyAISettings]):
>>>>>>> refs/remotes/origin/anmolMainAuto
    provider_name = ModelProviderName.UNIFYAI

    def __init__(self, settings: UnifyAISettings):
        super().__init__(settings)
        self.client = Unify(**settings.credentials.get_api_access_kwargs())
        self.async_client = AsyncUnify(**settings.credentials.get_api_access_kwargs())
        self.rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_tokens)

    def get_models(self) -> List[ChatModelInfo]:
        return list(UNIFY_AI_CHAT_MODELS.values())

    def get_embedding_models(self) -> List[EmbeddingModelInfo]:
        return list(UNIFY_AI_EMBEDDING_MODELS.values())

    async def create_chat_completion(
        self,
        prompt_messages: list[ChatMessage], # type: ignore if not in use 
        model: UnifyAiModelName,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[Dict[str, Any], Generator[ChatCompletionChunk, None, None]]:
        await self.rate_limiter.acquire()
        try:
            if stream:
                return self._stream_chat_completion(model, messages, **kwargs)
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            self.rate_limiter.update(self.count_tokens_from_messages(messages, model))
            return response
        except Exception as e:
            raise self._handle_api_error(e)

    async def _stream_chat_completion(
        self,
        model: UnifyAiModelName,
        messages: List[Dict[str, str]],
        **kwargs: Any
    ) -> Generator[ChatCompletionChunk, None, None]:
        try:
            async for chunk in self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            ):
                yield chunk
                self.rate_limiter.update(len(chunk.choices[0].delta.content or ""))
        except Exception as e:
            raise self._handle_api_error(e)

    async def create_embedding(
        self,
        model: UnifyAiModelName,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        await self.rate_limiter.acquire()
        try:
            response = await self.async_client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            self.rate_limiter.update(self.count_tokens(text, model))
            return response['data'][0]['embedding']
        except Exception as e:
            raise self._handle_api_error(e)

    def count_tokens(self, text: str, model: Optional[UnifyAiModelName] = None) -> int:
        # This is a placeholder and should be replaced with actual implementation
        if model is None:
            raise ValueError("Model must be specified to count tokens.")

        # Tokenization logic based on the model type
        if model in UNIFY_AI_EMBEDDING_MODELS:
            # Use a basic word tokenizer for embedding models
            tokens = self._simple_tokenizer(text)
        elif model in UnifyChatModels:
            # Use a GPT-like tokenizer for chat models
            tokens = self._gpt_like_tokenizer(text)
        elif model in UnifyAiModelName:
            # Use a custom tokenizer for other models
            tokens = self._custom_tokenizer(text, model)
        else:
            raise ValueError(f"Unsupported model: {model}")
        return len(tokens)

    def count_tokens_from_messages(self, messages: List[Dict[str, str]], model: UnifyAiModelName) -> int:
        # Implement token counting logic for messages here
        # This is a placeholder and should be replaced with actual implementation
        """
        Count the number of tokens in a list of messages based on the specified model's tokenizer.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries containing 'role' and 'content'.
            model (UnifyAiModelName): The model for which to count tokens.

        Returns:
            int: The total number of tokens in all messages.
        """
        if model is None:
            raise ValueError("Model must be specified to count tokens.")

        total_tokens = 0

        for message in messages:
            if 'content' not in message:
                raise ValueError("Each message must have a 'content' field.")
                
            content = message['content']

            # Tokenization logic based on the model type
            if model in UNIFY_AI_EMBEDDING_MODELS:
                tokens = self._simple_tokenizer(content)
            elif model in UnifyChatModels:
                tokens = self._gpt_like_tokenizer(content)
            else:
                raise ValueError(f"Unsupported model: {model}")

            total_tokens += len(tokens)

        return total_tokens
        # return sum(len(message['content'].split()) for message in messages)

    @staticmethod
    def default_model() -> UnifyAiModelName:
        return UnifyAiModelName.MIXTRAL_8X7B_INSTRUCT_V1

    def _handle_api_error(self, e: Exception) -> UnifyAIError:
        if isinstance(e, UnifyAIError):
            return e
        elif "rate limit" in str(e).lower():
            return UnifyAIRateLimitError("Rate limit exceeded", code="rate_limit_exceeded")
        else:
            return UnifyAIError(f"Unexpected error: {str(e)}")

class UnifyAiSettings(ModelProviderSettings):
    credentials: Optional[UnifyAICredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore

class RateLimiter:
    def _init_(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_tokens = 0
        self.last_reset_time = time.time()

    async def acquire(self):
        current_time = time.time()
        if current_time - self.last_reset_time >= 60:
            self.request_tokens = 0
            self.last_reset_time = current_time
        
        if self.request_tokens >= self.requests_per_minute:
            await asyncio.sleep(60 - (current_time - self.last_reset_time))
            await self.acquire()
        
        self.request_tokens += 1

    def update(self, tokens: int):
        self.request_tokens += tokens // self.tokens_per_minute

class UnifyAIModelSelector:
    @staticmethod
    def select_model(task: str, input_length: int, quality_preference: str, budget: float) -> UnifyAiModelName:
        if task == "code" and input_length > 1000:
            return UnifyAiModelName.CODELLAMA_70B_INSTRUCT
        elif quality_preference == "high" and budget > 0.1:
            return UnifyAiModelName.GPT_4_TURBO
        elif quality_preference == "medium" and budget > 0.05:
            return UnifyAiModelName.CLAUDE_3_OPUS
        else:
            return UnifyAiModelName.MIXTRAL_8X7B_INSTRUCT_V1

    @staticmethod
    def estimate_cost(model: UnifyAiModelName, input_length: int) -> float:
        model_info = UnifyChatModels.get(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")
        
        estimated_tokens = input_length // 4  # Rough estimate, adjust based on actual tokenization
        return (estimated_tokens * model_info.prompt_token_cost) + \
               (estimated_tokens * model_info.completion_token_cost)

async def fine_tune_model(self, model: UnifyAiModelName, training_data: List[Dict[str, str]]) -> str:
    # This is a placeholder for fine-tuning functionality
    # Implement actual fine-tuning logic based on UnifyAI's API
    try:
        response = await self.async_client.fine_tunes.create(
            model=model,
            training_data=training_data
        )
        return response['id']
    except Exception as e:
        raise self._handle_api_error(e)

async def get_fine_tune_status(self, fine_tune_id: str) -> Dict[str, Any]:
    #have to write logic related to fine tuning status
    try:
        response = await self.async_client.fine_tunes.retrieve(fine_tune_id)
        return response
    except Exception as e:
        raise self._handle_api_error(e)