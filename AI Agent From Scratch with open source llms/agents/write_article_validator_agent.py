from agent_base import AgentBase
# idea: search other tool for validation as well 
class WriterValidatorAgent(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="WriterValidatorAgent", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, topic, article):
        # system prompt
        system_message= "You are an expert AI Assistant that validates research articles"
        user_content=(
            "Given the topic and article, asses whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
            "Provide brief analysis and rate the article on scale of 1 to 10, where 10 indiates excellent quality\n\n"
            f"Topic: {topic}\n\n"
            f"Article:\n{article}"
            "Validation: "
        )

        messages=[
            {"role": "system", "content":system_message },
            {"role": "user", "content": user_content}
            
            ]
        validation= self.call_llm(messages, max_tokens=512)
        return validation
    
    
        