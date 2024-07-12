# type: ignore
expected_plans = {
    "Scan all corporate filings for LIPO and notify me of any big developments or changes to cash flow": [
        '[{"tool_name": "stock_identifier_lookup", "tool_task_id": "0c1cd887-c157-4447-a462-06af316822e6", '
        '"args": {"stock_name": "LIPO"}, "description": "Get the stock identifier for LIPO", "output_variable_name": '
        '"lipo_stock_id", "is_output_node": false, "store_output": true}, {"tool_name": "get_10k_10q_sec_filings", '
        '"tool_task_id": "7570173f-80aa-4d0f-8589-7fd1ca98cde1", "args": {"stock_ids": [{"var_name": '
        '"lipo_stock_id"}]},'
        '"description": "Get the 10-K and 10-Q SEC filings for LIPO", "output_variable_name": "sec_filings", '
        '"is_output_node": false, "store_output": false}, {"tool_name": "summarize_texts", "tool_task_id": '
        '"3ea6364b-84e8-42fd-a7e2-6152a9fc2fe2", "args": {"texts": {"var_name": "sec_filings"}, "topic": "big '
        'developments or changes to cash flow"}, "description": "Summarize big developments or changes to cash flow '
        'in the filings", "output_variable_name": "summary", "is_output_node": false, "store_output": true}, '
        '{"tool_name": "prepare_output", "tool_task_id": "47249bf4-603c-489d-af37-48cb1717277d", '
        '"args": {"object_to_output": {"var_name": "summary"}, "title": "Summary of Big Developments or Changes to '
        'Cash Flow in LIPO\'s Corporate Filings"}, "description": "Output the summary", "output_variable_name": '
        '"output", "is_output_node": true, "store_output": false}]',
        '[{"tool_name": "stock_identifier_lookup", "tool_task_id": "04220121-6ba8-45c1-a550-ea1347d1ed8f", '
        '"args": {"stock_name": "LIPO"}, "description": "Get the stock identifier for LIPO", "output_variable_name": '
        '"stock_id", "is_output_node": false, "store_output": true}, {"tool_name": "get_date_range", "tool_task_id": '
        '"2ec277ec-b3b6-43c4-a9da-74b63c47f0a3", "args": {"date_range_str": "last year"}, "description": "Get the '
        'date range for the last year to scan filings from that period", "output_variable_name": "date_range", '
        '"is_output_node": false, "store_output": true}, {"tool_name": "get_all_text_data_for_stocks", '
        '"tool_task_id": "f66742b2-1898-4224-bc75-b497ee460fba", "args": {"stock_ids": [{"var_name": "stock_id"}], '
        '"date_range": {"var_name": "date_range"}}, "description": "Fetch all text data including SEC filings for '
        'LIPO within the last year", "output_variable_name": "all_texts", "is_output_node": false, "store_output": '
        'false}, {"tool_name": "summarize_texts", "tool_task_id": "16866e48-364d-4378-9d59-4c1b9ce2e1fb", '
        '"args": {"texts": {"var_name": "all_texts"}, "topic": "big developments or changes to cash flow"}, '
        '"description": "Summarize the texts to find big developments or changes to cash flow", '
        '"output_variable_name": "summary", "is_output_node": false, "store_output": true}, {"tool_name": '
        '"prepare_output", "tool_task_id": "f52e0e1b-b88f-41ab-8a68-593900226852", "args": {"object_to_output": {'
        '"var_name": "summary"}, "title": "Summary of Big Developments or Changes to Cash Flow for LIPO"}, '
        '"description": "Display the summarized developments or changes to cash flow", "output_variable_name": '
        '"final_output", "is_output_node": true, "store_output": false}]',
    ]
}
