from pydantic import BaseModel, Field


class NewsHeadline(BaseModel):
    label: list[float] = Field(
        description="Topic distribution across four categories in the following order: world, sports, business, science/technology. Values should sum to 1."
        )
    text: str = Field(
        description="Generated news headline that reflects the topic distribution."
    )


class NewsHeadlineCoT(NewsHeadline):
    reasoning: str = Field(
        description="Full step-by-step reasoning process necessary to write the news headline."
    )
