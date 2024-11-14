FROM 374053208103.dkr.ecr.us-west-2.amazonaws.com/boosted-python-base:0.0.19
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
# install pandoc for document conversion
RUN apt-get install -y pandoc
RUN apt-get install -y libpq-dev
RUN pip install pipenv==2024.0.1
WORKDIR /service
COPY Pipfile ./
ENV CARGO_BUILD_JOBS=1
ENV PIPENV_MAX_SUBPROCESS=2
RUN pipenv install --verbose
COPY agent_service/ ./agent_service
COPY application.py .
COPY no_auth_endpoints.py .
COPY definitions.py .
COPY regression_test/ ./regression_test
COPY prefect_sqs_serve.py .
COPY sqs_execute.py .
COPY cron_scheduler_worker.py .
COPY quality_agent_ingestion_worker.py .
COPY scripts/ ./scripts
COPY config/ ./config
CMD ["pipenv", "run", "python", "application.py"]
