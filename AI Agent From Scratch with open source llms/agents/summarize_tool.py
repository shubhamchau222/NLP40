# all the logics for summarization will be in this file
# jina ai, crwal ai, etc for crawl purpose

from agent_base import AgentBase

class SummerizationTool(AgentBase):
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="SummarizeTool", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, text):
        # system prompt
        messages=[
                        {"role": "system", 
                        "content": "You are an AI Assistant that summarize medical texts."},

                        {"role":"user",
                        "content": ("Please provide a consise summary of the following medical text:\n\n"
                                    f"{text}\n\nSummary:" 
                                    )
                        
                        }
                ]
        
        # call llm & pass prompt with text
        summary= self.call_llm(messages, max_tokens=300)
        return summary
    
        
        
