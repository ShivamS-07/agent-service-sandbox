# Submitting jobs to Prefect DEV from local

To submit a job to Prefect DEV from local, you need to configure the environment variable in your terminal: `export PREFECT_API_URL=http://prefect-dev.boosted.ai:4200/api`

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

# For running document conversion endpoint locally
Install pandoc with homebrew: `brew install pandoc`.


# Test End-to-End locally
In order to test the end-to-end flow locally, you need to have the following setup:
1. Set environment variable `export REDIS_QUEUE_HOST=boosted-redis-celery.5vekgx.ng.0001.usw2.cache.amazonaws.com`
2. Launch the `agent-service` by running `python application.py` in the `agent-service` directory. 
3. Launch the `agent-web` by running `npm run dev:local` in the `agent-web` directory.