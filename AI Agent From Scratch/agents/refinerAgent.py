from .agent_base import AgentBase

class RefinerAgent(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="RefinerAgent", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, draft):
        messages=[
                        {"role": "system", 
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an expert editor who refines and enhances articles for clarity, coherence and academic quality. "
                            }
                        ]},

                        {"role":"user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please refine the following article draft to imrove its language, coherence, and overall quality/:\n\n"
                                f"{draft}\n\nRefinedArticle:"
                            }
                        ]
                        
                        }
                ]
   
        Refinedarticle= self.call_llm(messages, temperature=0.4,max_tokens=1000)
        return Refinedarticle
    