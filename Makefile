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

