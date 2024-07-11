# type: ignore
from agent_service.io_type_utils import IOType
from regression_test.test_regression import (
    TestExecutionPlanner,
    get_output,
    validate_plan,
)
from regression_test.util import compare_with_expected_text


class TestMarketCommentary(TestExecutionPlanner):

    def test_market_commentary_past_month(self):
        prompt = "Write a commentary on market performance for the month of Jan 2024"

        def validate_output(prompt: str, output: IOType):
            output_text = get_output(output=output)
            expected_text = (
                "January 2024 has been a month of significant market movements, driven by a confluence of "
                "geopolitical tensions, economic data releases, and central bank actions. The performance "
                "of various asset classes, particularly gold, has been influenced by these factors, "
                "reflecting the broader economic landscape.\n**Introduction**\nThe start of 2024 has been "
                "marked by heightened volatility in the financial markets. Key drivers include "
                "geopolitical tensions, particularly the ongoing conflict between Russia and Ukraine, "
                "and the anticipation of central bank actions, especially from the U.S. Federal Reserve. "
                "These factors have had a profound impact on asset prices, including gold, which has seen "
                "significant fluctuations throughout the month.\n**Geopolitical Tensions and Their "
                "Impact**\nOne of the most significant developments in January 2024 has been the "
                "escalation of the Russia-Ukraine conflict. The conflict has entered a new phase, "
                "with both sides targeting energy assets to disrupt each other's economies. Ukrainian "
                "drone attacks on Russian oil refineries and Russian strikes on Ukrainian energy "
                "infrastructure have created significant disruptions in global energy markets. This has "
                "led to increased volatility in oil prices, which have risen due to concerns about supply "
                "disruptions (Text 280, Text 199).\nThe geopolitical tensions have also had a direct "
                "impact on gold prices. As a safe-haven asset, gold typically benefits from increased "
                "geopolitical risks. Throughout January, gold prices have experienced significant "
                "fluctuations, driven by the ebb and flow of news from the conflict zone. For instance, "
                "gold prices surged following reports of Ukrainian drone strikes on Russian refineries, "
                "reflecting investor concerns about the potential for broader economic disruptions (Text "
                "280, Text 199).\n**Central Bank Actions and Economic Data**\nAnother major factor "
                "influencing market performance in January 2024 has been the actions and anticipated "
                "actions of central banks, particularly the U.S. Federal Reserve. The Fed's monetary "
                "policy has been a key driver of market sentiment, with investors closely watching for "
                "signals about future interest rate cuts. The anticipation of a potential rate cut in "
                "March has been a significant factor supporting gold prices, as lower interest rates "
                "reduce the opportunity cost of holding non-yielding assets like gold (Text 104, "
                "Text 268).\nEconomic data releases have also played a crucial role in shaping market "
                "performance. For example, stronger-than-expected U.S. employment data led to a temporary "
                "dip in gold prices, as it reduced the likelihood of an imminent rate cut by the Fed. "
                "However, subsequent data indicating a slowdown in inflation reignited hopes for a rate "
                "cut, leading to a rebound in gold prices (Text 119, Text 297).\n**Supply Chain "
                "Disruptions**\nSupply chain disruptions have continued to be a significant theme in "
                "January 2024. The collapse of the Francis Scott Key Bridge in Baltimore has caused major "
                "disruptions to the Port of Baltimore, one of the busiest ports in the U.S. This incident "
                "has had ripple effects across various industries, including the automotive sector, "
                "which relies heavily on the port for the import and export of vehicles and parts. The "
                "disruption has led to increased costs and delays, further complicating an already "
                "strained global supply chain (Text 302, Text 306).\n**Relevance to the Portfolio**\nFor "
                "investors with significant exposure to gold, the developments in January 2024 underscore "
                "the importance of monitoring geopolitical risks and central bank actions. The "
                "fluctuations in gold prices highlight the asset's role as a hedge against uncertainty "
                "and inflation. Investors should consider the potential for continued volatility in gold "
                "prices, driven by ongoing geopolitical tensions and central bank "
                "policies.\nAdditionally, the supply chain disruptions caused by the Baltimore bridge "
                "collapse serve as a reminder of the vulnerabilities in global logistics networks. "
                "Investors with exposure to industries reliant on complex supply chains, "
                "such as automotive and manufacturing, should be aware of the potential for increased "
                "costs and delays.\n**Conclusion**\nJanuary 2024 has been a month of significant market "
                "movements, driven by geopolitical tensions, central bank actions, and supply chain "
                "disruptions. The performance of gold and other assets reflects the broader economic "
                "landscape, characterized by uncertainty and volatility. Investors should remain "
                "vigilant, monitoring key developments and adjusting their strategies accordingly to "
                "navigate the complex and dynamic market environment."
            )
            self.loop.run_until_complete(
                compare_with_expected_text(
                    llm=self.llm,
                    output_text=output_text,
                    prompt=prompt,
                    expected_text=expected_text,
                )
            )

        self.prompt_test(
            prompt=prompt,
            validate_plan=validate_plan,
            validate_output=validate_output,
            raise_plan_validation_error=True,
        )
