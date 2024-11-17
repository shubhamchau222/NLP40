from .agent_base import AgentBase

class ArticleWriterTool(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="ArticleWriterTool", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, topic, outline=None):
        # system prompt
        system_message= "You are an expert academic writer."
        user_content=f"write research article on the following topic\nTopic: {topic}"

        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += f"Artivle:\n"

        messages=[
            {"role": "system", "content":system_message },
            {"role": "user", "content": user_content}
            
            ]
        article= self.call_llm(messages, max_tokens=1000)
        return article
    
    
        