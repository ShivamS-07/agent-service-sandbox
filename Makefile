uv:
	uv sync

pipenv: uv

clean:
	uv --rm || true
	uv install --dev --verbose

format:
	uv run invoke format

check:
	uv run invoke check

checkformat:
	uv run invoke checkformat


verify:
	uv run invoke verify

.PHONY: test
test:
	uv run invoke test


.PHONY: testfast
testfast:
	uv run invoke testfast

.PHONY: testslow
testslow:
	uv run invoke testslow

.PHONY: regtest
regtest:
	uv run invoke testregression
