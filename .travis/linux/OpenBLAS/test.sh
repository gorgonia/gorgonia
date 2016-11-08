set -ex

go env

go test -v -a -coverprofile=test.cover .
go test -v -a -coverprofile=./tensor/f64/test.cover ./tensor/f64
go test -v -a -coverprofile=./tensor/f32/test.cover ./tensor/f32
go test -v -a -coverprofile=./tensor/i/test.cover ./tensor/i
go test -v -a -coverprofile=./tensor/b/test.cover ./tensor/b
go test -tags='sse' -v -a  ./...
go test -tags='avx' -v -a  ./...


goveralls -coverprofile=./test.cover -service=travis-ci
goveralls -coverprofile=./tensor/f64/test.cover -service=travis-ci
goveralls -coverprofile=./tensor/f32/test.cover -service=travis-ci
goveralls -coverprofile=./tensor/i/test.cover -service=travis-ci
goveralls -coverprofile=./tensor/n/test.cover -service=travis-ci


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
