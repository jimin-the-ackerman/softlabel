from .imdb import prompt_template as imdb_prompt_template
from .imdb import details as imdb_details

from .sst import prompt_template as sst_prompt_template
from .sst import details as sst_details

from .subj import prompt_template as subj_prompt_template
from .subj import details as subj_details

from .emotion import prompt_template as emotion_prompt_template
from .emotion import details as emotion_details

from .agnews import prompt_template as agnews_prompt_template
from .agnews import details as agnews_details


data_to_template = {
    'imdb': imdb_prompt_template,
    'sst': sst_prompt_template,
    'subj': subj_prompt_template,
    'emotion': emotion_prompt_template,
    'agnews': agnews_prompt_template,
}

data_to_detailed_guidelines = {  # TODO: deprecate
    'imdb': imdb_details,
    'sst': sst_details,
    'subj': subj_details,
    'emotion': emotion_details,
    'agnews': agnews_details,
}