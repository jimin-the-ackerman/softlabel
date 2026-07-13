
from pydantic import BaseModel, Field


class MovieReview(BaseModel):
    label: float = Field(
        description="Sentiment score of the review (0 for negative, 1 for positive)"
    )
    text: str = Field(
        description="Final generated movie review"
    )


class MovieReviewCoT(MovieReview):
    reasoning: str = Field(
        description="Full step-by-step reasoning process necessary to write the final review"
    )
