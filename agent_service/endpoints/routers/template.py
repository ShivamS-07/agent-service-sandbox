from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from agent_service.endpoints.authz_helper import User, parse_header
from agent_service.endpoints.models import (
    CreatePromptTemplateRequest,
    CreatePromptTemplateResponse,
    DeletePromptTemplateRequest,
    DeletePromptTemplateResponse,
    FindTemplatesRelatedToPromptRequest,
    FindTemplatesRelatedToPromptResponse,
    GenPromptTemplateFromPlanRequest,
    GenPromptTemplateFromPlanResponse,
    GenTemplatePlanRequest,
    GenTemplatePlanResponse,
    GetCompaniesResponse,
    GetPromptTemplatesResponse,
    RunTemplatePlanRequest,
    RunTemplatePlanResponse,
    UpdatePromptTemplateRequest,
    UpdatePromptTemplateResponse,
)
from agent_service.endpoints.routers.utils import get_agent_svc_impl
from agent_service.utils.feature_flags import is_user_agent_admin

router = APIRouter(prefix="/api/template")


@router.get(
    "/get-all-companies", response_model=GetCompaniesResponse, status_code=status.HTTP_200_OK
)
async def get_all_companies(user: User = Depends(parse_header)) -> GetCompaniesResponse:
    agent_svc_impl = get_agent_svc_impl()
    is_user_admin = user.is_super_admin or await is_user_agent_admin(
        user.user_id, async_db=agent_svc_impl.pg
    )
    if not is_user_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access"
        )
    return await agent_svc_impl.get_all_companies(user)


@router.get(
    "/get-prompt-templates",
    response_model=GetPromptTemplatesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_prompt_templates(user: User = Depends(parse_header)) -> GetPromptTemplatesResponse:
    agent_svc_impl = get_agent_svc_impl()
    templates = await agent_svc_impl.get_prompt_templates(user)
    return GetPromptTemplatesResponse(prompt_templates=templates)


@router.post(
    "/create-prompt-template",
    response_model=CreatePromptTemplateRequest,
    status_code=status.HTTP_200_OK,
)
async def create_prompt_template(
    req: CreatePromptTemplateRequest, user: User = Depends(parse_header)
) -> CreatePromptTemplateResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.create_prompt_template(
        name=req.name,
        user=user,
        description=req.description,
        prompt=req.prompt,
        category=req.category,
        plan_run_id=req.plan_run_id,
        organization_ids=req.organization_ids,
        cadence_tag=req.cadence_tag,
        notification_criteria=req.notification_criteria,
    )


@router.post(
    "/generate-template-plan",
    response_model=GenTemplatePlanResponse,
    status_code=status.HTTP_200_OK,
)
async def gen_template_plan(
    req: GenTemplatePlanRequest, user: User = Depends(parse_header)
) -> GenTemplatePlanResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.gen_template_plan(template_prompt=req.template_prompt, user=user)


@router.post(
    "/run-template-plan", response_model=RunTemplatePlanResponse, status_code=status.HTTP_200_OK
)
async def create_agent_and_run_template(
    req: RunTemplatePlanRequest, user: User = Depends(parse_header)
) -> RunTemplatePlanResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.create_agent_and_run_template(
        template_prompt=req.template_prompt,
        notification_criteria=req.notification_criteria,
        plan=req.plan,
        is_draft=req.is_draft,
        cadence_description=req.cadence_description,
        user=user,
    )


@router.post(
    "/delete-template-prompt",
    response_model=DeletePromptTemplateResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_prompt_template(
    req: DeletePromptTemplateRequest, user: User = Depends(parse_header)
) -> DeletePromptTemplateResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.delete_prompt_template(template_id=req.template_id)


@router.post(
    "/update-template-prompt",
    response_model=UpdatePromptTemplateResponse,
    status_code=status.HTTP_200_OK,
)
async def update_prompt_template(
    req: UpdatePromptTemplateRequest, user: User = Depends(parse_header)
) -> UpdatePromptTemplateResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.update_prompt_template(
        template_id=req.template_id,
        name=req.name,
        description=req.description,
        category=req.category,
        prompt=req.prompt,
        plan=req.plan,
        cadence_tag=req.cadence_tag,
        notification_criteria=req.notification_criteria,
        organization_ids=req.organization_ids,
    )


@router.post(
    "/gen-template-from-plan",
    response_model=GenPromptTemplateFromPlanResponse,
    status_code=status.HTTP_200_OK,
)
async def gen_prompt_template_from_plan(
    req: GenPromptTemplateFromPlanRequest, user: User = Depends(parse_header)
) -> GenPromptTemplateFromPlanResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.gen_prompt_template_from_plan(
        plan_run_id=req.plan_run_id, agent_id=req.agent_id
    )


@router.post(
    "/find-templates-related-to-prompt",
    response_model=FindTemplatesRelatedToPromptResponse,
    status_code=status.HTTP_200_OK,
)
async def find_templates_related_to_prompt(
    req: FindTemplatesRelatedToPromptRequest, user: User = Depends(parse_header)
) -> FindTemplatesRelatedToPromptResponse:
    agent_svc_impl = get_agent_svc_impl()
    return await agent_svc_impl.find_templates_related_to_prompt(query=req.query, user=user)
