FROM python:3.12-slim
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
# install pandoc for document conversion
RUN apt-get install -y pandoc
RUN apt-get install -y libpq-dev
RUN pip install pipenv
WORKDIR /service
COPY Pipfile ./
RUN pipenv install --verbose
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
