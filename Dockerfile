FROM 374053208103.dkr.ecr.us-west-2.amazonaws.com/boosted-python-base:latest
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
RUN pip install pipenv==2022.3.28
WORKDIR /service
COPY Pipfile.lock Pipfile ./
RUN pipenv install
RUN pipenv run pip install "clickhouse-connect==0.7.18"
COPY agent_service/ ./agent_service
COPY application.py .
COPY no_auth_endpoints.py .
COPY definitions.py .
COPY regression_test/ ./regression_test
COPY prefect_serve.py .
COPY prefect_sqs_serve.py .
COPY sqs_execute.py .
COPY cron_scheduler_worker.py .
COPY scripts/ ./scripts
COPY config/ ./config
CMD ["pipenv", "run", "python", "application.py"]
