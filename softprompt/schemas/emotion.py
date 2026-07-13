from pydantic import BaseModel, Field


class Text(BaseModel):
    label: list[float] = Field(
        description="Emotion distribution across six emotions (precisely in the following order: sadness, joy, love, anger, fear, surprise)"
        )
    text: str = Field(description="Generated text")


class TextCoT(Text):
    reasoning: str = Field(
        description="Full step-by-step reasoning process necessary to write the text."
    )
