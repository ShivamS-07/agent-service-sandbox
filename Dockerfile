FROM 374053208103.dkr.ecr.us-west-2.amazonaws.com/boosted-python-base:0.0.19
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
# install pandoc for document conversion
RUN apt-get install -y pandoc
RUN apt-get install -y libpq-dev
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y gcc
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN ln -s /root/.local/bin/uv /usr/local/bin/pipenv
WORKDIR /service
COPY pyproject.toml .
RUN uv sync
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
CMD ["uv", "run", "python", "application.py"]
