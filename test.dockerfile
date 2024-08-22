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
RUN pipenv install --dev
COPY . .
ENV ENVIRONMENT=DEV
CMD ["pipenv", "run", "invoke", "verify"]
