from agent_base import AgentBase
# idea: search other tool for validation as well 
class SummarizeValidatorAgent(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="SummarizeValidatorAgent", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, original_text, summary):
        # system prompt
        system_message= "You are an expert AI Assistant that validates the summaries of medical texts."
        user_content=(
            "Given the original sumary, asses whether the summary accurately capture the key points and is of high quality.\n"
            "Provide brief analysis and rate the article on scale of 1 to 10, where 10 indiates excellent quality\n\n"
            f"Original Text: {original_text}\n\n"
            f"Summary:\n{summary} \n\n"
            "Validation: "
        )

        messages=[
            {"role": "system", "content":system_message },
            {"role": "user", "content": user_content}
            
            ]
        validation= self.call_llm(messages, max_tokens=512)
        return validation
    
    
        