import asyncio

from gbi_common_py_utils.utils.environment import DEV_TAG, PROD_TAG
from gbi_common_py_utils.utils.postgres import PostgresBase

from agent_service.GPT.constants import DEFAULT_EMBEDDING_MODEL
from agent_service.GPT.requests import GPT


async def populate_embedding_in_prompt_templates(environment: str) -> None:
    print(f"Fetching prompt templates from {environment} environment")
    db = PostgresBase(environment=environment, skip_commit=False)
    llm = GPT(model=DEFAULT_EMBEDDING_MODEL)

    prompt_templates = db.generic_read(
        """
        SELECT template_id, description, description_embedding
        FROM agent.prompt_templates"""
    )
    for template in prompt_templates:
        if not template["description_embedding"]:
            template["description_embedding"] = await llm.embed_text(template["description"])
            print(template["description_embedding"])
            sql = """
            UPDATE agent.prompt_templates
            SET description_embedding = %(description_embedding)s
            WHERE template_id = %(template_id)s
            """
            db.generic_write(
                sql,
                {
                    "template_id": template["template_id"],
                    "description_embedding": template["description_embedding"],
                },
            )


async def main() -> None:
    # populate embedding for PROD
    await populate_embedding_in_prompt_templates(PROD_TAG)

    # populate embedding for DEV
    await populate_embedding_in_prompt_templates(DEV_TAG)


if __name__ == "__main__":
    asyncio.run(main())
