package main

type ci struct {
	Workflows         []workflow
	StableGoVersion   string
	PreviousGoVersion string
}

type workflow struct {
	WorkflowName string
	WorkflowFile string
	Jobs         []job
}

type job struct {
	JobID     string          //stable-go
	JobName   string          //Build and test on latest stable Go release
	RunsOn    string          //ubuntu-latest
	GoVersion string          //1.15.x
	Tags      map[string]bool // none:false, avx:true, sse:true
	WithRace  bool            // false
}

const workflowTmpl = `## DO NOT EDIT - This file is generated
on: [pull_request]
name: {{ .WorkflowName }}
env:
  GOPROXY: "https://proxy.golang.org"
jobs:
{{- range .Jobs }}{{ template "Job" . }}{{ end }}

{{- define "Job" }}
  {{ .JobID }}:
    name: {{ .JobName }} 
    env:
      GOVERSION: '{{ .GoVersion }}'
    strategy:
      matrix:
        experimental: [false]
        tags: [{{ mapToList .Tags }}]
        include:
{{- range $tag,$experimental := .Tags}}
{{- if $experimental }}  
          - tags: {{ $tag }}
            experimental: true
{{- end}}
{{- end}}
    runs-on: "{{ .RunsOn }}"
    continue-on-error: ${{"{{"}} matrix.experimental {{"}}"}}
    steps:
    - name: Install Go 
      uses: actions/setup-go@v2
      with:
        go-version: ${{"{{"}} env.GOVERSION {{"}}"}}
    # Get values for cache paths to be used in later steps
    - id: go-cache-paths
      run: |
        echo "::set-output name=go-build::$(go env GOCACHE)"
        echo "::set-output name=go-mod::$(go env GOMODCACHE)"
    - name: Checkout
      uses: actions/checkout@v2
    # Cache go build cache, used to speedup go test
    - name: Go Build Cache
      if: steps.go-cache-paths.outputs.go-build != ''
      id: build-cache
      uses: actions/cache@v2
      with:
        path: ${{"{{"}} steps.go-cache-paths.outputs.go-build {{"}}"}}
        key: ${{"{{"}} runner.os {{"}}"}}-go-build-${{"{{"}} hashFiles('**/go.sum') {{"}}"}}
        restore-keys: |
          ${{"{{"}} runner.os {{"}}"}}-go-build- 
    # Cache go mod cache, used to speedup builds
    - name: Go Mod Cache
      if: steps.go-cache-paths.outputs.go-mod != ''
      id: build-mod-cache
      uses: actions/cache@v2
      with:
        path: ${{"{{"}} steps.go-cache-paths.outputs.go-mod {{"}}"}}
        key: ${{"{{"}} runner.os {{"}}"}}-go-mod-${{"{{"}} hashFiles('**/go.sum') {{"}}"}}
        restore-keys: |
          ${{"{{"}} runner.os {{"}}"}}-go-mod- 
    - name: Install Dependencies
      if: steps.build-mod-cache.outputs.cache-hit != 'true'
      run: |
        GOARCH=arm GOOS=linux go get -t .
        GOARCH=amd64 GOOS=linux go get -t .
        GOARCH=amd64 GOOS=darwin go get -t .
    - name: Build without tags (all plateforms)
      if: matrix.tags == 'none'
      run: |
        GOARCH=arm GOOS=linux go build . 
        GOARCH=amd64 GOOS=linux go build .
        GOARCH=amd64 GOOS=darwin go build .
    - name: Test without tags
      if: matrix.tags == 'none'
      run: |
        go test {{if .WithRace}}-race{{end}}-timeout 20m
    - name: Build with tag 
      if: matrix.tags != 'none'
      run: |
        go build -tags=${{"{{"}} matrix.tags {{"}}"}}
    - name: Test with tag
      if: matrix.tags != 'none'
      run: |
        go test {{if .WithRace}}-race{{end}} -timeout 20m -tags=${{"{{"}} matrix.tags {{"}}"}}
{{- end }}
`
