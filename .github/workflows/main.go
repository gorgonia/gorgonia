package main

import (
	"io"
	"os"
	"strings"
	"text/template"
)

const (
	latestGo   = "1.15.x"
	previousGo = "1.14.x"
)

func main() {
	workflowLinux, err := os.OpenFile("runner-github-ubuntu-amd64.yml", os.O_RDWR|os.O_CREATE, 0660)
	if err != nil {
		panic(err)
	}
	defer workflowLinux.Close()
	workflowMac, err := os.OpenFile("runner-github-macos-amd64.yml", os.O_RDWR|os.O_CREATE, 0660)
	if err != nil {
		panic(err)
	}
	defer workflowMac.Close()
	workflowSelf, err := os.OpenFile("runner-self-hosted.yml", os.O_RDWR|os.O_CREATE, 0660)
	if err != nil {
		panic(err)
	}
	defer workflowSelf.Close()

	err = generateWorkflow(workflowLinux, "Build and Tests on Linux/amd64", "Linux/amd64", "ubuntu-latest", map[string]bool{
		"none": false,
		"avx":  false,
		"sse":  false,
	}, true)
	if err != nil {
		panic(err)
	}
	err = generateWorkflow(workflowMac, "Build and Tests on MacOS/amd64", "MacOS/amd64", "macos-latest", map[string]bool{
		"none": false,
		"avx":  false,
		"sse":  false,
	}, false)
	if err != nil {
		panic(err)
	}
	err = generateWorkflow(workflowSelf, "Build and Tests on Self-Hosted (arm)", "Self-Hosted", "self-hosted", map[string]bool{
		"none": false,
		"avx":  true,
		"sse":  true,
	}, false)
	if err != nil {
		panic(err)
	}

}

func generateWorkflow(w io.Writer, workflowName, runnerName, runsOn string, tags map[string]bool, withRace bool) error {
	tmpl, err := template.New("workflow").Funcs(template.FuncMap{
		"mapToList":       mapToList,
		"hasExperimental": hasExperimental,
	}).Parse(workflowTmpl)
	if err != nil {
		panic(err)
	}

	return tmpl.Execute(w, workflow{
		WorkflowName: workflowName,
		Jobs: []job{
			{
				JobID:     "stable-go",
				JobName:   "Build and test on latest stable Go release - " + runnerName,
				RunsOn:    runsOn,
				GoVersion: latestGo,
				Tags:      tags,
				WithRace:  withRace,
			},
			{
				JobID:     "previous-go",
				JobName:   "Build and test on previous stable Go release - " + runnerName,
				RunsOn:    runsOn,
				GoVersion: previousGo,
				Tags:      tags,
				WithRace:  withRace,
			},
		},
	})
}

func mapToList(m map[string]bool) string {
	var b strings.Builder
	for tag := range m {
		b.WriteString(tag)
		b.WriteString(",")
	}
	s := b.String()   // no copying
	s = s[:b.Len()-1] // no copying (removes trailing ", ")
	return s

}

func hasExperimental(m map[string]bool) bool {
	for _, ok := range m {
		if ok {
			return true
		}
	}
	return false
}
