import unittest

from agent_service.agent_quality_worker.jira_integration import JiraIntegration
from agent_service.endpoints.models import JiraTicketCriteria


class TestJiraIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up JiraIntegration instance for all tests
        cls.jira_integration = JiraIntegration()

    def test_create_ticket_success(self):
        """
        Test creating a Jira ticket and transitioning it to 'Done' status.
        """
        # Step 1: Define criteria for the Jira ticket
        criteria = JiraTicketCriteria(
            project_key="QL24",
            summary="Automated Ticket Creation",
            description="This ticket was created through the API with custom criteria.",
            issue_type="Bug",
            assignee="william.ngana@gradientboostedinvestments.com",
            priority="High",
            labels=["automation", "python"],
        )

        # Step 2: Attempt to create a ticket and verify the response
        issues = self.jira_integration.create_ticket(criteria)
        self.assertIsNotNone(issues, "Failed to create ticket - ticket ID is None")

        # Extract the ticket key
        ticket_key = issues[0].get("key")
        self.assertIsNotNone(ticket_key, "Ticket key is None - creation might have failed")
        ticket_url = issues[0].get("self")
        self.assertIsNotNone(ticket_url, "Ticket URL is None - creation might have failed")

        # Step 3: Delete the ticket
        self.jira_integration.delete_ticket(ticket_key=ticket_key)
