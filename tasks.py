from invoke import task


@task
def checkformat(c):
    c.run("ruff check")  # flake8
    c.run("ruff check --select I")  # isort
    c.run("ruff format --check")  # black


@task
def check(c):
    checkformat(c)
    mypy(c)
    pyanalyze(c)


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
        "RUN_IN_CI=true uv run python -m pytest  --durations=0 --durations-min=5.0"
        " --disable-warnings --capture=tee-sys  --log-cli-level=WARNING"
        " -n 16 --dist worksteal"
        " -v tests"
    )


@task
def test(c):
    testfast(c)
    testregressionci(c)


# ci versions only run parts of the reg test that are for each PR
@task
def testregressionci(c):
    c.run(
        "RUN_IN_CI=true uv run python -m pytest  --durations=0 --durations-min=5.0"
        " --disable-warnings --capture=tee-sys  --log-cli-level=WARNING"
        " -n 8 --dist worksteal"
        " -v regression_test"
    )


@task
def testregressionslowci(c):
    c.run("RUN_IN_CI=true python -W ignore -m unittest discover -v -s regression_test")


# run these to invoke the full regression tests
@task
def testregression(c):
    c.run(
        "uv run python -m pytest -n 32 --dist worksteal -v regression_test --log-level=CRITICAL"
        " --durations=0 --durations-min=5.0"
    )


@task
def testregressionslow(c):
    c.run("python -W ignore -m unittest discover -v -s regression_test")


@task
def format(c):
    c.run("ruff check --fix")  # flake8
    c.run("ruff check --select I --fix")  # isort
    c.run("ruff format")  # black


@task
def coverage(c):
    c.run("coverage run -m unittest discover -v")
    c.run("coverage xml -o coverage.xml")


def check_unpublished_protobuf_version():
    with open("pyproject.toml") as f:
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


pyanalyze_files_dirs = [
    "agent_service/tools/",
    "agent_service/utils/",
]


@task
def pyanalyze(c):
    # mypy Not detecting UnboundLocalError / uninitialized vars / doesn't analyze branches
    # https://github.com/python/mypy/issues/2400#issuecomment-798477686

    # https://github.com/quora/pyanalyze
    # there will be some false positives, often it is better to pretend they are real
    # and just declare & initialize the variable higher up
    # but you can also use this comment to ignore a line: # static analysis: ignore
    print("pyanalyze")
    pyanalyze_files_str = " ".join(pyanalyze_files_dirs)
    errors = [
        "possibly_undefined_name",
        "missing_await",
    ]

    errors_str = "-e " + " -e ".join(errors)
    c.run(f"python -m pyanalyze --disable-all {errors_str} {pyanalyze_files_str}")
