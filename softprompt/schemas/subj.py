
from pydantic import BaseModel, Field


class Example(BaseModel):
    label: float = Field(
        description="Subjectivity score of the text (0 for objective, 1 for subjective)"
    )
    text: str = Field(
        description="Final generated text"
    )


class ExampleCoT(Example):
    reasoning: str = Field(
        description="Full step-by-step reasoning process necessary to write the text"
    )