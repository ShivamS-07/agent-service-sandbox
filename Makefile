pipenv:
	pipenv install --dev --verbose

clean:
	pipenv --rm
	pipenv install --dev --verbose

format:
	pipenv run invoke format

check:
	pipenv run invoke check

checkformat:
	pipenv run invoke checkformat


verify:
	pipenv run invoke verify

.PHONY: test
test:
	pipenv run invoke test


.PHONY: testfast
testfast:
	pipenv run invoke testfast

.PHONY: testslow
testslow:
	pipenv run invoke testslow

.PHONY: regtest
regtest:
	pipenv run invoke testregression
