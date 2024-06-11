from collections import defaultdict
from typing import Dict, List, Tuple

from data_access_layer.core.dao.securities import SecuritiesMetadataDAO

from agent_service.io_types.text import Text, ThemeText
from agent_service.tools.commentary.constants import (
    MAX_ARTICLES_PER_DEVELOPMENT,
    MAX_DEVELOPMENTS_PER_TOPIC,
)
from agent_service.tools.themes import (
    GetThemeDevelopmentNewsArticlesInput,
    GetThemeDevelopmentNewsInput,
    get_news_articles_for_theme_developments,
    get_news_developments_about_theme,
)
from agent_service.types import PlanRunContext


async def get_sec_metadata_dao() -> SecuritiesMetadataDAO:
    return SecuritiesMetadataDAO(cache_sec_metadata=True)


# Helper functions


async def get_theme_related_texts(
    themes_texts: List[ThemeText], context: PlanRunContext
) -> List[Text]:
    """
    This function gets the theme related texts for the given themes.
    """
    # print("themes texts size", len(themes_texts))
    res: List = []
    development_texts = await get_news_developments_about_theme(
        GetThemeDevelopmentNewsInput(
            themes=themes_texts, max_devs_per_theme=MAX_DEVELOPMENTS_PER_TOPIC
        ),
        context,
    )
    # print("development texts size", len(development_texts))
    article_texts = await get_news_articles_for_theme_developments(  # type: ignore
        GetThemeDevelopmentNewsArticlesInput(
            developments_list=development_texts,  # type: ignore
            max_articles_per_development=MAX_ARTICLES_PER_DEVELOPMENT,
        ),
        context,  # type: ignore
    )
    # print("article texts size", len(article_texts))
    res.extend(development_texts)  # type: ignore
    res.extend(article_texts)  # type: ignore
    return res


async def organize_commentary_texts(texts: List[Text]) -> Tuple[List[Text], List[Text], List[Text]]:
    """
    This function organizes the commentary texts by themes, developments and articles.
    """
    themes: List[Text] = []
    developments: List[Text] = []
    articles: List[Text] = []
    for text in texts:
        if text.text_type == "Theme Description":
            themes.append(text)
        elif text.text_type == "News Development Summary":
            developments.append(text)
        else:
            articles.append(text)
    return themes, developments, articles


async def get_portfolio_geography_prompt(regions_to_weight: List[Tuple[str, float]]) -> str:
    # convert weights to int percentages
    portfolio_geography = "\n".join(
        [f"{tup[0]}: {int(tup[1] * 100)}%" for tup in regions_to_weight]
    )
    return portfolio_geography


async def get_region_weights_from_portfolio_holdings(
    weighted_holdings: Dict[int, float]
) -> List[Tuple[str, float]]:
    """
    Given a mapping from GBI ID to a weight, return a list of ranked (region,
    weight) tuples sorted in descending order by weight.
    """
    dao = await get_sec_metadata_dao()
    sec_meta_map = dao.get_security_metadata(list(weighted_holdings.keys())).get()
    region_weight_map: Dict[str, float] = defaultdict(float)
    for meta in sec_meta_map.values():
        region_weight_map[meta.country] += weighted_holdings[meta.gbi_id]

    output = list(region_weight_map.items())
    return sorted(output, reverse=True, key=lambda tup: tup[1])