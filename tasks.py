from invoke import task


@task
def checkformat(c):
    c.run("black --check .")
    c.run("isort --profile black --check .")
    c.run("flake8")


@task
def check(c):
    checkformat(c)
    mypy(c)


@task
def verify(c):
    check(c)
    test(c)


@task
def testslow(c):

    print("running tests slowly serially")
    c.run("RUN_IN_CI=true python -W ignore -m unittest discover -v -s tests")


@task
def testfast(c):
    # runs each test class in its own process 8 at a time
    print("running tests in parallel")
    c.run(
        "RUN_IN_CI=true unittest-parallel -v --level class --disable-process-pooling --jobs 3  "
        "-t . -s tests"
    )


@task
def test(c):
    testfast(c)
    testregression(c)


@task
def testregression(c):
    c.run(
        "RUN_IN_CI=true unittest-parallel -v --level class --disable-process-pooling --jobs 3  "
        "-t . -s regression_test"
    )


@task
def format(c):
    c.run("isort .")
    c.run("black .")
    c.run("flake8 .")


@task
def coverage(c):
    c.run("coverage run -m unittest discover -v")
    c.run("coverage xml -o coverage.xml")


def check_unpublished_protobuf_version():
    with open("Pipfile") as f:
        # pa-portfolio-service-proto-v1 = {version = "==0.0.0+8df543afc7b50c992dfeb25f89187529453c8520", index = "gbi"} # noqa
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if "proto" in line and "==0.0.0+" in line:
                print(f"unpublished protobuf version detected: {line}")
                return True

    return False


@task
def mypy(c):
    print("mypy")
    disable_for_unpublished_proto_changes = ""
    if check_unpublished_protobuf_version():
        disable_for_unpublished_proto_changes = (
            "--disable-error-code attr-defined --disable-error-code unused-ignore"
        )
        print(f"will run mypy with {disable_for_unpublished_proto_changes}")

    c.run(
        f"mypy --no-incremental {disable_for_unpublished_proto_changes} --config-file pyproject.toml ."
    )
