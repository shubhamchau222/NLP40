from agent_base import AgentBase

class SanitizeDataTool(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="SanitizeTool", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, text_data):
        # system prompt
        messages=[
                        {"role": "system", 
                        "content": "You are an AI Assistant that sanitizes medical data by removing protected health information (PHI)."},

                        {"role":"user",
                        "content": ("Remove all PHI from the following data:\n\n"
                                    f"{text_data}\n\nSanitize Data:" 
                                    )
                        
                        }
                ]
        
        # call llm & pass prompt with text
        SanitizeData= self.call_llm(messages, max_tokens=300)
        return SanitizeData
    
    
        