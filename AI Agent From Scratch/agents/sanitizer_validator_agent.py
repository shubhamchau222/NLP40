from .agent_base import AgentBase
# idea: search other tool for validation as well 
class SanitizeValidatorAgent(AgentBase):
    """ This tool will remove all Protected Health Information from given medical text Data
    """
    def __init__(self, max_retrieves=2, verbose=True):
        super().__init__(name="SanitizeValidatorAgent", max_retrieves=max_retrieves, verbose=verbose)
    
    def execute(self, original_data, sanitize_data):
        # system prompt
        system_message= "You are an expert AI Assistant that validates the sanitization of medical data by checking the removal of Personal Health Information (PHI)."
        user_content=(
            "Given the original data and sanitized data, verify that all (PHI) data is completely removed.\n"
            "Provide brief analysis and rate the article on scale of 1 to 10, where 10 indiates excellent quality\n\n"
            f"Original Data: {original_data}\n\n"
            f"Sanitize Data:\n{sanitize_data} \n\n"
            "Validation: "
        )

        messages=[
            {"role": "system", "content":system_message },
            {"role": "user", "content": user_content}
            
            ]
        validation= self.call_llm(messages, max_tokens=512)
        return validation
    
    
        