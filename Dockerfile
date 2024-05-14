FROM 374053208103.dkr.ecr.us-west-2.amazonaws.com/boosted-python-base:latest
# add security updates
RUN apt-get update
RUN apt-get -s dist-upgrade | grep "^Inst" | grep -i securi | awk -F " " {'print $2'} | xargs apt-get install
RUN pip install pipenv==2022.3.28
WORKDIR /service
COPY Pipfile.lock Pipfile ./
RUN pipenv install
COPY agent_service/ ./agent_service
COPY application.py .
CMD ["pipenv", "run", "python", "application.py"]
