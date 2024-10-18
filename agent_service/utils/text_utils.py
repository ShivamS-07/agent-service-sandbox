from typing import List

from agent_service.io_types.text import (
    StockDescriptionSectionText,
    StockDescriptionText,
    StockEarningsSummaryPointText,
    StockEarningsSummaryText,
    StockEarningsTranscriptSectionText,
    StockEarningsTranscriptText,
    StockOtherSecFilingSectionText,
    StockOtherSecFilingText,
    StockSecFilingSectionText,
    StockSecFilingText,
    Text,
)
from agent_service.types import PlanRunContext


async def partition_to_smaller_text_sizes(texts: List[Text], context: PlanRunContext) -> List[Text]:
    # If more texts get partition functionality they can be added here
    sec_filing_texts: List[StockSecFilingText] = []
    other_sec_filing_texts: List[StockOtherSecFilingText] = []
    earning_summary_texts: List[StockEarningsSummaryText] = []
    earning_transcript_texts: List[StockEarningsTranscriptText] = []
    company_description_texts: List[StockDescriptionText] = []

    partitioned_texts: List[Text] = []

    for text in texts:
        if isinstance(text, StockOtherSecFilingText):
            other_sec_filing_texts.append(text)
        elif isinstance(text, StockSecFilingText):
            sec_filing_texts.append(text)
        elif isinstance(text, StockEarningsSummaryText):
            earning_summary_texts.append(text)
        elif isinstance(text, StockEarningsTranscriptText):
            earning_transcript_texts.append(text)
        elif isinstance(text, StockDescriptionText):
            company_description_texts.append(text)

        else:
            partitioned_texts.append(text)

    if earning_transcript_texts:
        partitioned_transcript_texts = (
            await StockEarningsTranscriptSectionText.init_from_full_text_data(
                earning_transcript_texts, context
            )
        )
        partitioned_texts.extend(partitioned_transcript_texts)

    if earning_summary_texts:
        partitioned_earning_summary_texts = (
            await StockEarningsSummaryPointText.init_from_full_text_data(earning_summary_texts)
        )
        partitioned_texts.extend(partitioned_earning_summary_texts)

    if sec_filing_texts:
        partitioned_filings_texts = await StockSecFilingSectionText.init_from_full_text_data(
            sec_filing_texts
        )
        partitioned_texts.extend(partitioned_filings_texts)

    if other_sec_filing_texts:
        partitioned_other_filings_texts = (
            await StockOtherSecFilingSectionText.init_from_full_text_data(other_sec_filing_texts)
        )
        partitioned_texts.extend(partitioned_other_filings_texts)

    if company_description_texts:
        partitioned_description_texts = await StockDescriptionSectionText.init_from_full_text_data(
            company_description_texts
        )
        partitioned_texts.extend(partitioned_description_texts)

    return partitioned_texts
