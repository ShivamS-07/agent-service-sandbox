from fastapi import HTTPException, status

from agent_service.external.pa_svc_client import (
    delete_watchlist,
    delete_workspace,
    get_all_holdings_in_workspace,
    get_watchlist_stocks,
    rename_watchlist,
    rename_workspace,
)
from agent_service.io_type_utils import TableColumnType
from agent_service.io_types.output import Output
from agent_service.io_types.table import (
    STOCK_ID_COL_NAME_DEFAULT,
    TableOutput,
    TableOutputColumn,
)
from agent_service.types import MemoryType
from agent_service.utils.stock_metadata import get_stock_metadata


class MemoryHandler:
    async def get_content(self, user_id: str, id: str) -> Output:
        raise NotImplementedError("Subclasses must implement this method.")

    async def rename(self, user_id: str, id: str, new_name: str) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")

    async def delete(self, user_id: str, id: str) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")


def get_handler(type: str) -> MemoryHandler:
    handler: MemoryHandler
    if type == MemoryType.PORTFOLIO:
        handler = PortfolioMemoryHandler()
    elif type == MemoryType.WATCHLIST:
        handler = WatchlistMemoryHandler()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"type {type} is not supported"
        )
    return handler


class PortfolioMemoryHandler(MemoryHandler):
    async def get_content(self, user_id: str, id: str) -> TableOutput:
        workspace = await get_all_holdings_in_workspace(user_id, id)
        gbi_ids = [holding.gbi_id for holding in workspace.holdings]
        gbi_id2_stock = await get_stock_metadata(gbi_ids)
        stock_data = [gbi_id2_stock[gbi_id] for gbi_id in gbi_ids]
        rows = [
            [
                stock_data[i],
                workspace.holdings[i].weight,
            ]
            for i in range(len(stock_data))
        ]
        columns = [
            TableOutputColumn(name=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
            TableOutputColumn(name="Weight", col_type=TableColumnType.FLOAT),
        ]
        return TableOutput(title="Memory Content - Portfolio", columns=columns, rows=rows)  # type: ignore

    async def rename(self, user_id: str, id: str, new_name: str) -> bool:
        return await rename_workspace(user_id, id, new_name)

    async def delete(self, user_id: str, id: str) -> bool:
        return await delete_workspace(user_id=user_id, workspace_id=id)


class WatchlistMemoryHandler(MemoryHandler):
    async def get_content(self, user_id: str, id: str) -> TableOutput:
        gbi_ids = await get_watchlist_stocks(user_id, id)
        gbi_id2_stock = await get_stock_metadata(gbi_ids)
        stock_data = [gbi_id2_stock[gbi_id] for gbi_id in gbi_ids]
        rows = [[stock] for stock in stock_data]
        columns = [
            TableOutputColumn(name=STOCK_ID_COL_NAME_DEFAULT, col_type=TableColumnType.STOCK),
        ]
        return TableOutput(title="Memory Content - Watchlist", columns=columns, rows=rows)  # type: ignore

    async def rename(self, user_id: str, id: str, new_name: str) -> bool:
        return await rename_watchlist(user_id, id, new_name)

    async def delete(self, user_id: str, id: str) -> bool:
        return await delete_watchlist(user_id=user_id, watchlist_id=id)
