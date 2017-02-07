package main

import (
	"io"
	"reflect"
	"text/template"
)

type testData struct {
	Kind      reflect.Kind
	TestData0 []interface{}
}

func makeTests(generic *ManyKinds) []testData {
	retVal := make([]testData, 0)
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			td := testData{
				Kind: k,
			}

			data := make([]interface{}, 5)
			if isRangeable(k) {
				raw := []int{0, 1, 2, 3, 4}
				for i := range data {
					data[i] = raw[i]
				}
			} else if k == reflect.Bool {
				raw := []bool{true, false, true, false, true}
				for i := range data {
					data[i] = raw[i]
				}
			} else if k == reflect.String {
				raw := []string{"\"zero\"", "\"one\"", "\"two\"", "\"three\"", "\"four\""}
				for i := range data {
					data[i] = raw[i]
				}
			} else {
				continue
			}
			td.TestData0 = data
			retVal = append(retVal, td)
		}
	}
	return retVal
}

const getTestRaw = `var denseSetGetTests = []struct {
	of Dtype
	data interface{} 

	correct []interface{}
}{
	{{range . -}}
	{{$k := .Kind -}}
	{ {{title .Kind.String | strip}}, []{{.Kind.String | clean}}{ {{range .TestData0 -}}{{printf "%v" .}}, {{end -}} }, []interface{}{ {{range .TestData0 -}} {{$k}}({{printf "%v" .}}), {{end -}} }},
	{{end -}}
}

func TestDense_setget(t *testing.T) {
	assert := assert.New(t)
	for _, gts := range denseSetGetTests {
		T := New(Of(gts.of), WithShape(len(gts.correct)))
		for i, v := range gts.correct {
			T.Set(i, v)
			got := T.Get(i)
			assert.Equal(v, got)
		}
	}
}

`

var (
	GetTest *template.Template
)

func init() {
	GetTest = template.Must(template.New("GetTest").Funcs(funcs).Parse(getTestRaw))
}

func getsetTest(f io.Writer, generic *ManyKinds) {
	tests := makeTests(generic)
	GetTest.Execute(f, tests)
}
