set -ex

go env

go test -v -a -coverprofile=test.cover .
go test -v -a -coverprofile=./tensor/test.cover ./tensor/

go test -tags='sse' -v -a  ./...
go test -tags='avx' -v -a  ./...

# because coveralls only accepts one coverage file at one time... we combine them into one gigantic one
covers=(./test.cover ./tensor/test.cover)
echo "mode: set" > ./final.cover
tail -q -n +2 "${covers[@]}" >> ./final.cover
goveralls -coverprofile=./final.cover -service=travis-ci

#if [[ $TRAVIS_SECURE_ENV_VARS = "true" ]]; then bash -c "$GOPATH/src/github.com/$TRAVIS_REPO_SLUG/.travis/test-coverage.sh"; fi

# travis compiles commands in script and then executes in bash.  By adding
# set -e we are changing the travis build script's behavior, and the set
# -e lives on past the commands we are providing it.  Some of the travis
# commands are supposed to exit with non zero status, but then continue
# executing.  set -x makes the travis log files extremely verbose and
# difficult to understand.
# 
# see travis-ci/travis-ci#5120
set +ex
