from pydantic import BaseModel, Field


class Review(BaseModel):
    label: float = Field(
        description="Sentiment score of the review (0 for negative, 1 for positive)"
        )
    text: str = Field(
        description="Final generated movie review"
    )


class ReviewCoT(Review):
    reasoning: str = Field(
        description="Full step-by-step reasoning process necessary to write the final review"
    )
