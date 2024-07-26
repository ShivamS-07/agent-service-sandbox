# Submitting jobs to Prefect DEV from local

To submit a job to Prefect DEV from local, you need to configure the environment variable in your terminal: `EXPORT PREFECT_API_URL=http://prefect-dev.boosted.ai:4200/api`

# Local Setup Doc
https://gradientboostedinvestments.atlassian.net/wiki/spaces/GBI/pages/2847145988/Agent+Web+Agent+Service+Local+Setup

# Formatting & Type Check
To format you files, run: `make format`\
To check types, run: `make check`

# Note
Before deployment, to sync the `agent_service.sample_plans` table, run the following from the /scripts directory:

```chatinput
python sample_plans_upsert.py --action backup_and_upsert 
```

This saves a copy of the current prod table under /backups and sync's the dev table with prod.
