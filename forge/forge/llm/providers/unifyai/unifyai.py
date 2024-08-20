import openai
import requests
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
            name=UnifyAiModelName.llama_3_8b_chat@together-ai,
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





class UnifyAIProvider(BaseOpenAIChatProvider[UnifyAIModelName, UnifyAISettings],
    BaseOpenAIEmbeddingProvider[UnifyAIModelName, UnifyAISettings],):
    provider_name = ModelProviderName.UNIFYAI

class UnifyAiSettings(ModelProviderSettings):
    credentials: Optional[UnifyAICredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore

