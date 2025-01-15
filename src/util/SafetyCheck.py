from pydantic import BaseModel, Field


class SafetyCheck(BaseModel):
    """Checks if a given input contains inhumane, illegal, or unethical content."""

    input: str = Field(description="The phrase, question, or demand to be classified.")
    is_unethical: bool = Field(
        default=False,
        description="True if the input contains unethical content. Unethical content includes, but is not limited to, mentions of violence, self-harm, illegal activity, lawbreaking, harm or danger to others, terrorism, or anything intended to cause injury or suffering."
    )
