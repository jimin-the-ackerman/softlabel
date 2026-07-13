"""
Structured output schemas for ProGen framework.
"""

from pydantic import BaseModel, Field
from typing import List


class MovieReviewGeneration(BaseModel):
    """Schema for movie review generation."""
    text: str = Field(description="The generated movie review text")
    label: float = Field(description="The label you were requested to generate (i.e., 0 is negative, 1 is positive)")

class MovieReviewGenerationCoT(MovieReviewGeneration):
    """Schema for movie review generation with Chain of Thought."""
    reasoning: str = Field(description="Full step-by-step reasoning process necessary to write the final review")

class SentenceGeneration(BaseModel):
    """Schema for sentence generation."""
    text: str = Field(description="The generated sentence text")
    label: float = Field(description="The label you were requested to generate (i.e., 0 is negative, 1 is positive)")

class SentenceGenerationCoT(SentenceGeneration):
    """Schema for sentence generation with Chain of Thought."""
    reasoning: str = Field(description="Full step-by-step reasoning process necessary to write the final sentence")

class TextGeneration(BaseModel):
    """Schema for general text generation."""
    text: str = Field(description="The generated text")
    label: float = Field(description="The label you were requested to generate (i.e., 0 is objective, 1 is subjective)")

class TextGenerationCoT(TextGeneration):
    """Schema for general text generation with Chain of Thought."""
    reasoning: str = Field(description="Full step-by-step reasoning process necessary to write the final text")
    
class EmotionGeneration(BaseModel):
    """Schema for emotion text generation."""
    text: str = Field(description="The generated text")
    # label: List[float] = Field(description="The label you were requested to generate")
    label: list[float] = Field(
        description="Emotion distribution across six emotions (precisely in the following order: sadness, joy, love, anger, fear, surprise)"
        )

class EmotionGenerationCoT(EmotionGeneration):
    """Schema for emotion text generation with Chain of Thought."""
    reasoning: str = Field(description="Full step-by-step reasoning process necessary to write the final text")


def get_output_schema(dataset: str, cot: bool = False):
    """Get the appropriate output schema for a dataset."""
    schemas = {
        "imdb": MovieReviewGenerationCoT if cot else MovieReviewGeneration,
        "sst": SentenceGenerationCoT if cot else SentenceGeneration,
        "subj": TextGenerationCoT if cot else TextGeneration,
        "emotion": EmotionGenerationCoT if cot else EmotionGeneration
    }
    return schemas[dataset] 