from pydantic import BaseModel
from .imdb import MovieReview, MovieReviewCoT
from .sst import Review, ReviewCoT
from .subj import Example, ExampleCoT
from .emotion import Text, TextCoT
from .agnews import NewsHeadline, NewsHeadlineCoT


data_to_output_schema = {
    'imdb': MovieReview,
    'sst': Review,
    'subj': Example,
    'emotion': Text,
    'agnews': NewsHeadline,
}


data_to_output_schema_cot = {
    'imdb': MovieReviewCoT,
    'sst': ReviewCoT,
    'subj': ExampleCoT,
    'emotion': TextCoT,
    'agnews': NewsHeadlineCoT,
}


def get_output_schema(data: str, cot: bool = False) -> BaseModel:
    if cot:
        return data_to_output_schema_cot[data]
    return data_to_output_schema[data]
    