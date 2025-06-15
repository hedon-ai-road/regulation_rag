import warnings
warnings.filterwarnings('ignore')
from langchain.llms.base import LLM
from typing import Any, List, Optional, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

class RagLLM(LLM):
    client: Optional[OpenAI] = None
    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        response = self.client.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct:free",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024
        )
        return  response.choices[0].message.content



    @property
    def _llm_type(self) -> str:
        return "rag_llm_qwen/qwen-2.5-72b-instruct:free"
    

from langchain_community.embeddings import HuggingFaceEmbeddings
class RagEmbedding(object):
    def __init__(self, model_name="BAAI/bge-m3",
                 device="cpu"):
        self.embedding = HuggingFaceEmbeddings(model_name=model_name,
                                               model_kwargs={"device": device})
    def get_embedding_fun(self):
        return self.embedding