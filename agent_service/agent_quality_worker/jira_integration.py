import logging
from typing import Any, Dict, List, Optional

from atlassian import Jira
from gbi_common_py_utils.utils.ssm import get_param

from agent_service.agent_quality_worker.constants import JIRA_URL, JIRA_USER
from agent_service.endpoints.models import JiraTicketCriteria


class JiraIntegration:
    """
    A class to handle Jira integration for creating tickets.

    Attributes:
        jira (Jira): An instance of the Jira client from the atlassian-python-api.
    """

    def __init__(self, url: str = JIRA_URL, user: str = JIRA_USER):
        """
        Initialize the JiraIntegration with authentication details.

        Args:
            url (str): The base URL of the Jira instance.
        """
        try:
            api_token = get_param("/jira/apikey")
            self.jira = Jira(url=url, username=user, password=api_token)
        except Exception as e:
            raise e

    def create_ticket(self, criteria: JiraTicketCriteria) -> Optional[List[Dict[str, Any]]]:
        """
        Creates a Jira ticket using the criteria provided in the JiraTicketCriteria model.

        Args:
            criteria (JiraTicketCriteria): The details for the Jira ticket.

        Returns:
            str: The ID of the created ticket, or None if creation fails.
        """
        # Check that mandatory fields are provided
        mandatory_fields = ["project_key", "summary", "description"]
        missing_fields = [
            field for field in mandatory_fields if getattr(criteria, field, None) is None
        ]

        if missing_fields:
            raise ValueError(f"Missing mandatory fields: {', '.join(missing_fields)}")

        # Dynamically construct the fields dictionary
        fields = {
            "project": {"key": criteria.project_key},
            "summary": criteria.summary,
            "description": criteria.description,
            "issuetype": {"name": criteria.issue_type or "Task"},
        }

        # Add any additional custom fields
        if criteria.custom_fields:
            fields.update(criteria.custom_fields)

        try:
            # Create the issue in Jira
            issues = self.jira.create_issues([{"fields": fields}])
            return issues.get("issues", [])
        except Exception as e:
            logging.warning(f"Error creating Jira ticket: {e}")
            return None

    def delete_ticket(self, ticket_key: str) -> None:
        try:
            self.jira.delete_issue(issue_id_or_key=ticket_key)
        except Exception as e:
            print(f"Error deleting Jira ticket: {e}")
