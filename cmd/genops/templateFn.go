package main

import (
	"strings"
	"text/template"
)

var funcmap = template.FuncMap{
	"lower": strings.ToLower,
	"title": strings.Title,
}
