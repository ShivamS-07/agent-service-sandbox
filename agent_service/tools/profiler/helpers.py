import json
from asyncio.log import logger
from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from agent_service.GPT.constants import GPT4_O
from agent_service.GPT.requests import GPT
from agent_service.GPT.tokens import GPTTokenizer
from agent_service.tools.profiler.constants import (
    GREATER_THAN_AVERAGE,
    IMPORTANCE_POSTFIX,
    MAX_INDUSTRIES,
    MIN_INDUSTRIES,
    MIN_INDUSTRY_RATING,
    NEGATIVE,
    NUM_IMPACT_INDUSTRY_TABLES,
    POSITIVE,
    REPEAT_EFFECT,
)
from agent_service.tools.profiler.prompts import (
    PROFILE_TABLE_MAIN,
    PROFILE_TABLE_SYS,
    SINGLE_PROFILE_MAIN,
    SINGLE_PROFILE_SYS,
)
from agent_service.utils.async_utils import gather_with_concurrency
from agent_service.utils.postgres import get_psql
from agent_service.utils.string_utils import clean_to_json_if_needed


@dataclass
class ImpactInfo:
    name: str
    description: str
    score: float
    # profiles added later
    profiles: Optional[Dict[str, List[Any]]] = None


def get_industries_for_impact(
    mean_df: pd.DataFrame,
    sorted_industry_idxs: np.ndarray,
    done_industry_idxs: List[int],
    wanted: np.ndarray,
) -> List[str]:
    # this iterates over the ranked list corresponding to one column of the table (an impact), creates
    # and returns a final list of industries for the impact as long as they are wanted and the min
    # max are satisfied. Returns the list of industries, the indexs of added industries are also
    # added to done_industry_idxs
    found_industries: List[str] = []
    j = 1
    while len(found_industries) < MAX_INDUSTRIES and (
        wanted[sorted_industry_idxs[-j]] or len(found_industries) < MIN_INDUSTRIES
    ):
        index = sorted_industry_idxs[-j]
        if index not in done_industry_idxs:
            done_industry_idxs.append(index)
        found_industries.append(mean_df.index[index])
        j += 1
    return found_industries


def get_industries_for_impacts(dfs: List[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    # this function takes a list of panadas dataframes corresponding to scores relating the
    # industries (rows) to impacts (columns, each dataframe is from a separate run of GPT
    # We average the scores together, and then impact by impact we identify those industries
    # which have a high (and noticeably higher than average) score for that impact. We
    # start with impacts with low overall averages (which tend to be the ones which just
    # one or two relevant industries), and as we assign relevant industries we try to avoid
    # reusing these industries unless we have to, to improve the diversity of the theme overall
    # we return a dict mapping impacts to dictionaries of industries to score mappings
    mean_df = cast(pd.DataFrame, sum(dfs) / len(dfs))  # have to cast because mypy
    impact_avgs = mean_df.mean(axis=0).to_numpy()
    done_industry_idxs: List[int] = []
    output = {}
    for i in np.argsort(impact_avgs):  # iterate over impacts
        curr_avg = impact_avgs[i]
        impact = mean_df.columns[i]
        data = mean_df[impact].to_numpy()
        data[done_industry_idxs] += REPEAT_EFFECT
        wanted = (data > curr_avg + GREATER_THAN_AVERAGE) & (data > MIN_INDUSTRY_RATING)
        sorted_industry_idxs = np.argsort(data)
        industries = get_industries_for_impact(
            mean_df, sorted_industry_idxs, done_industry_idxs, wanted
        )
        output[impact] = {industry: mean_df.loc[industry, impact] for industry in industries}
    return output


async def write_industry_impact_table(
    theme: str, impacts: List[Dict[str, str]], text_data: str
) -> pd.DataFrame:
    db = get_psql()
    llm = GPT(model=GPT4_O)
    industries_from_db = db.get_industry_names()
    industries_from_db.remove("Media")  # duplicate Media
    industries = "\n".join(industries_from_db)

    # Very rare but seems that sometimes a column is skipped
    tries = 0
    while tries < 3:
        chopped_texts_str = GPTTokenizer(model=llm.model).do_truncation_if_needed(
            truncate_str=text_data,
            other_prompt_strs=[
                PROFILE_TABLE_MAIN.template,
                PROFILE_TABLE_SYS.template,
                theme,
                str(impacts),
            ],
        )
        result = await llm.do_chat_w_sys_prompt(
            main_prompt=PROFILE_TABLE_MAIN.format(
                theme=theme,
                impacts=impacts,
                text_documents=chopped_texts_str,
                industries=industries,
            ),
            sys_prompt=PROFILE_TABLE_SYS.format(),
        )
        df = pd.read_csv(StringIO(result.strip("`'\"\n")), sep="\t", index_col=0)

        if set([impact["name"] for impact in impacts]) == set(df.columns):
            break
        else:
            tries += 1
    return df


async def write_impact_profiles(
    theme: str, impact: Dict[str, str], industries: List[str], impacts_str: str
) -> Dict[str, List[str]]:
    llm = GPT(model=GPT4_O)

    result = await llm.do_chat_w_sys_prompt(
        main_prompt=SINGLE_PROFILE_MAIN.format(
            theme=theme, impact=impact, impacts=impacts_str, industries=industries
        ),
        sys_prompt=SINGLE_PROFILE_SYS.format(),
    )
    try:
        profile_dict = json.loads(clean_to_json_if_needed(result))
        if (
            isinstance(profile_dict, dict)
            and len(set(profile_dict.keys()) - set([POSITIVE, NEGATIVE])) == 0
        ):
            return profile_dict
        else:
            return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to load profile generation output, error: {e}, impact: {impact}")
        return {}


def add_profile_importance(
    profiles: Dict[str, List[str]], industry_scores: Dict[str, float]
) -> Dict[str, List[Any]]:
    # this adds two new optional fields to the profiles which stores the scores associated with
    # the industries for each profile. An industry is matched to a profile using string matching
    # (GPT should have inserted an industry into each profile). There is a new field for each
    # of positive and negative, the lists of scores are the same length as the lists of profiles
    # This helps keeps things backwards compatible
    new_profile_dict: Dict[str, List[Any]] = {}
    for polarity, profile_list in profiles.items():
        new_profile_dict[polarity] = profile_list
        importance_list = []
        for profile in profile_list:
            final_score = MIN_INDUSTRY_RATING
            for industry, score in industry_scores.items():
                if industry in profile:
                    final_score = score
            importance_list.append(final_score)
        new_profile_dict[polarity + IMPORTANCE_POSTFIX] = importance_list

    return new_profile_dict


async def write_profiles(
    theme: str, impacts: List[Dict[str, str]], news: str
) -> Dict[str, Dict[str, List[Any]]]:
    # writing profiles proceeds in two steps, both with concurrency. First, GPT quantifies the
    # relationships between industries and impacts, and we extract a set of relevant industries
    # for each impact. Then, for each impact, GPT writes a specific set of profiles based on
    # the selected industries
    tasks = []
    for _ in range(NUM_IMPACT_INDUSTRY_TABLES):
        tasks.append(write_industry_impact_table(theme, impacts, news))
    dfs = await gather_with_concurrency(tasks)
    impact_industry_dict = get_industries_for_impacts(dfs)
    impacts_str = "\n".join([impact["name"] for impact in impacts])
    tasks = []
    for impact in impacts:
        impacted_industries = impact_industry_dict.get(impact["name"])
        if impacted_industries:
            tasks.append(
                write_impact_profiles(theme, impact, list(impacted_industries), impacts_str)
            )
        else:
            logger.warning(
                f"Failed to find impacted industries for '{impact["name"]}' during profile generation!"
            )

    all_profiles = await gather_with_concurrency(tasks)
    output = {}
    for impact, profiles in zip(impacts, all_profiles):
        output[impact["name"]] = add_profile_importance(
            profiles, impact_industry_dict[impact["name"]]
        )
    return output


def has_profile(profiles: Dict[str, List[str]]) -> bool:
    return any([profile for profile_list in profiles.values() for profile in profile_list])


def convert_impact_to_info(impact: Dict[str, Any]) -> ImpactInfo:
    return ImpactInfo(
        name=impact["name"],
        description=impact["description"],
        score=impact["score"],
        profiles=impact["profiles"],
    )
