set -ex

go env

go test -v . 
go test -v ./tensor/f64
go test -v ./tensor/f32
go test -v ./tensor/i
go test -v ./tensor/b
go test -tags='sse' -v . 
go test -tags='sse' -v ./tensor/f64
go test -tags='sse' -v ./tensor/f32
go test -tags='avx' -v . 
go test -tags='avx' -v ./tensor/f64
go test -tags='avx' -v ./tensor/f32

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
