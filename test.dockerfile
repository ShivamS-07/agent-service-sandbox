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
COPY . .
ENV ENVIRONMENT=DEV
CMD ["uv", "run", "invoke", "verify"]
