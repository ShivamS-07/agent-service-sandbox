import json
import logging
import os
from typing import Optional

from gbi_common_py_utils.utils.environment import (
    DEV_TAG,
    PROD_TAG,
    STAGING_TAG,
    VALID_ENVIRONMENTS,
    get_environment_tag,
    get_instance_id,
    get_tag_value,
    running_in_jenkins,
)

logger = logging.getLogger(__name__)

LOCAL_MACHINE = "LOCAL_MACHINE"
IS_SANDBOX = "IS_SANDBOX"


def _get_aws_ssm_prefix(env_tag: str) -> str:
    """
    Converts the given environment into a SSM prefix (i.e. 'dev')
    Returns: The SSM prefix used to get parameters for.

    """
    ssm_prefix = DEV_TAG.lower()
    if env_tag not in VALID_ENVIRONMENTS:
        raise Exception("Unknown Environment - {}".format(env_tag))
    elif env_tag == STAGING_TAG or env_tag == PROD_TAG:
        ssm_prefix = env_tag.lower()
    return ssm_prefix


def _get_s3_prefix(ssm_prefix: str) -> str:
    """
    Converts a SSM prefix (i.e. 'dev') into S3 bucket prefix
    """
    if ssm_prefix == "dev":
        return ""
    else:
        return "-" + ssm_prefix


def _get_llm_config(llm_config_file: Optional[str]) -> dict:
    if llm_config_file is None:
        return {}
    try:
        with open(llm_config_file, "r") as file:
            return json.load(file)  # type: ignore
    except Exception as e:
        raise Exception("Unable to retrieve llm configuration from file") from e


class EnvironmentUtils:
    """
    This class deals with managing environment state.
    """

    # get_environment_tag() will raise a RuntimeException if the user sets an invalid ENVIRONMENT
    # tag, otherwise defaults to LOCAL if not set. Valid tags are in VALID_ENVIRONMENTS.
    environment = get_environment_tag()
    # By default we will run with LOCAL for feature branching.

    # The instance-ID that the program is being run on.
    instance_id = LOCAL_MACHINE if get_instance_id() is None else get_instance_id()
    # If the machine is running in jenkins.
    is_jenkins = running_in_jenkins()
    # If the machine is running on sandbox.
    is_sandbox = False if get_tag_value(IS_SANDBOX) is None else True
    # If the machine is running locally.
    is_local = instance_id == LOCAL_MACHINE
    # Environment is "deployed" if it is not LOCAL or SANDBOX.
    # This is equivalent to pa_utils.is_running_on_aws()
    is_deployed = not is_jenkins and not is_sandbox and not is_local
    # The AWS SSM prefix (i.e. 'dev', 'alpha')
    aws_ssm_prefix = _get_aws_ssm_prefix(env_tag=environment)
    # The S3 PA bucket prefix (i.e. '', '-alpha')
    s3_pa_prefix = _get_s3_prefix(aws_ssm_prefix)
    # If true, then unit tests are being run.
    is_testing = False
    # Use job config if specified
    llm_config = _get_llm_config(os.environ.get("LLM_CONFIG_FILE"))

    @classmethod
    def re_initialize(cls, environment: str) -> None:
        cls.environment = environment
        cls.aws_ssm_prefix = _get_aws_ssm_prefix(env_tag=environment)
        cls.s3_pa_prefix = _get_s3_prefix(cls.aws_ssm_prefix)
