import openai
from abc import ABC, abstractmethod # abstract method
from loguru import logger
import os 
from dotenv import load_dotenv

# here we'll define our llm
openai.api_key= os.getenv("OPENAI_API_KEY")

## tools list


class AgentBase(ABC):
    def __init__(self, name, max_retrieves=2, verbose=True):
        super().__init__()
        """
        name: Agent Name
        max_retrives= If llm not able to generate op in one go, then how many re-tries it should make
        verbose= to give internal insight
        """
        self.name= name
        self.max_retrieves= max_retrieves
        self.verbose = verbose
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_llm(self, messages, temperature=0.7, max_tokens=150):
        retries=0
        while retries < self.max_retrieves:
            try:
                if self.verbose:
                    logger.info(f"[{self.name} sending message to LLM]")
                    for msg in messages:
                        logger.debug(f"{msg['role']}: {msg['content']}")
                response= openai.chat.completions.create(
                    model="gpt-40",
                    messages= messages,
                    temperature = temperature,
                    max_tokens = max_tokens
                )
                reply= response.choises[0].message
                if self.verbose:
                    logger.info(f"[{self.name} Received Response: {reply}]")
                return reply 
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name} Error during OpenAI Calls: {e} .. retry {retries}/{self.max_retrieves}]")
        raise Exception(f"[{self.name}] Failed to get response from OpenAI after {self.max_retrieves} retries. ")


            


