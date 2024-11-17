# from .refinerAgent import RefinerAgent
from refinerAgent import RefinerAgent
from sanitize_tool import SanitizeDataTool
from sanitizer_validator_agent import SanitizeValidatorAgent
from summarize_tool import SummerizationTool
from summary_validator_agent import SummarizeValidatorAgent
from write_article_validator_agent import WriterValidatorAgent
from write_articletool import ArticleWriterTool
from validator_agent import ValidatorAgent

class AgentManager:
    """Manage Entry points"""
    def __init__(self, max_retrieves=2, verbose=True):
        self.agents= {
        "summarize": SummerizationTool(max_retrieves=max_retrieves, verbose=verbose),
        "write_article": ArticleWriterTool(max_retrieves=max_retrieves, verbose=verbose),
        "sanitize_data": SanitizeDataTool(max_retrieves=max_retrieves, verbose=verbose),
        "summarize_validation": SummarizeValidatorAgent(max_retrieves=max_retrieves, verbose=verbose),
        "write_articlevalidator": WriterValidatorAgent(max_retrieves=max_retrieves, verbose=verbose),
        "sanitize_validator": SanitizeValidatorAgent(max_retrieves=max_retrieves, verbose=verbose),
        "refiner": RefinerAgent(max_retrieves=max_retrieves, verbose=verbose),      # New agent
        "validator": ValidatorAgent(max_retrieves=max_retrieves, verbose=verbose)   # New agent
            }
        
    def get_agent(self, agent_name):
        agent= self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"{agent_name}: Agent not found")
        return agent
    


