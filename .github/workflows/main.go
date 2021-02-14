package main

import (
	"os"
	"strings"
	"text/template"
)

const (
	latestGo   = "1.15.x"
	previousGo = "1.14.x"
)

func main() {
	tmpl, err := template.New("workflow").Funcs(template.FuncMap{"mapToList": mapToList}).Parse(workflowTmpl)
	if err != nil {
		panic(err)
	}
	err = tmpl.Execute(os.Stdout, workflow{
		WorkflowName: "Build and Tests on Linux/amd64",
		Jobs: []job{
			{
				JobID:     "stable-go",
				JobName:   "Build and test on latest stable Go release",
				RunsOn:    "ubuntu-latest",
				GoVersion: latestGo,
				Tags: map[string]bool{
					"none": false,
					"avx":  true,
					"sse":  true,
				},
				WithRace: true,
			},
			{
				JobID:     "previous-go",
				JobName:   "Build and test on previous stable Go release",
				RunsOn:    "ubuntu-latest",
				GoVersion: previousGo,
				Tags: map[string]bool{
					"none": false,
					"avx":  true,
					"sse":  true,
				},
				WithRace: true,
			},
		},
	})
	if err != nil {
		panic(err)
	}
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
