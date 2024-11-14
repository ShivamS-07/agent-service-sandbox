FROM 374053208103.dkr.ecr.us-west-2.amazonaws.com/boosted-python-base:0.0.19
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
# install pandoc for document conversion
RUN apt-get install -y pandoc
RUN apt-get install -y libpq-dev
RUN pip install pipenv==2024.0.1
WORKDIR /service
ENV CARGO_BUILD_JOBS=1
ENV PIPENV_MAX_SUBPROCESS=2
COPY Pipfile ./
RUN pipenv install --verbose
RUN pipenv install --dev
COPY . .
ENV ENVIRONMENT=DEV
CMD ["pipenv", "run", "invoke", "verify"]
