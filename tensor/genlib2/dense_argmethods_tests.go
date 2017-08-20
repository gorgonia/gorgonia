package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type ArgMethodTestData struct {
	Kind reflect.Kind
	Data []int
}

var data = []int{
	3, 4, 2, 4, 3, 8, 3, 9, 7, 4, 3, 0, 3, 9, 9, 0, 6, 7, 3, 9, 4, 8, 5,
	1, 1, 9, 4, 0, 4, 1, 6, 6, 4, 9, 3, 8, 1, 7, 0, 7, 4, 0, 6, 8, 2, 8,
	0, 6, 1, 6, 2, 3, 7, 5, 7, 3, 0, 8, 6, 5, 6, 9, 7, 5, 6, 8, 7, 9, 5,
	0, 8, 1, 4, 0, 6, 6, 3, 3, 8, 1, 1, 3, 2, 5, 9, 0, 4, 5, 3, 1, 9, 1,
	9, 3, 9, 3, 3, 4, 5, 9, 4, 2, 2, 7, 9, 8, 1, 6, 9, 4, 4, 1, 8, 9, 8,
	0, 9, 9, 4, 6, 7, 5, 9, 9, 4, 8, 5, 8, 2, 4, 8, 2, 7, 2, 8, 7, 2, 3,
	7, 0, 9, 9, 8, 9, 2, 1, 7, 0, 7, 9, 0, 2, 4, 8, 7, 9, 6, 8, 3, 3, 7,
	2, 9, 2, 8, 2, 3, 6, 0, 8, 7, 7, 0, 9, 0, 9, 3, 2, 6, 9, 5, 8, 6, 9,
	5, 6, 1, 8, 7, 8, 1, 9, 9, 3, 7, 7, 6, 8, 2, 1, 1, 5, 1, 4, 0, 5, 1,
	7, 9, 5, 6, 6, 8, 7, 5, 1, 3, 4, 0, 1, 8, 0, 2, 6, 9, 1, 4, 8, 0, 5,
	6, 2, 9, 4, 4, 2, 4, 4, 4, 3,
}

const argMethodsDataRaw = `var basicDense{{short .Kind}} = New(WithShape(2,3,4,5,2), WithBacking([]{{asType .Kind}}{ {{range .Data -}}{{.}}, {{end -}} }))
`

const argmaxCorrect = `var argmaxCorrect = []struct {
	shape Shape
	data  []int
}{
	{Shape{3,4,5,2}, []int{
		1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
		1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
		1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
		1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
		0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
		1, 0, 0, 0, 0,
	}},
	{Shape{2,4,5,2}, []int{
		1, 0, 1, 1, 2, 0, 2, 0, 0, 1, 2, 1, 2, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
		2, 2, 0, 1, 1, 2, 2, 1, 0, 2, 0, 2, 0, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 2, 1, 0, 1, 2, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 2, 1, 0, 1, 0,
		0, 2, 1, 1, 0, 0, 0, 0, 0, 2, 0,
	}},
	{Shape{2,3,5,2}, []int{
		3, 2, 2, 1, 1, 2, 1, 0, 0, 1, 3, 2, 1, 0, 1, 0, 2, 2, 3, 0, 1, 0, 1,
		3, 0, 2, 3, 3, 2, 1, 2, 2, 0, 0, 1, 3, 2, 0, 1, 2, 0, 3, 0, 1, 0, 1,
		3, 2, 2, 1, 2, 1, 3, 1, 2, 0, 2, 2, 0, 0,
	}},
	{Shape{2,3,4,2}, []int{
		4, 3, 2, 1, 1, 2, 0, 1, 1, 1, 1, 3, 1, 0, 0, 2, 2, 1, 0, 4, 2, 2, 3,
		1, 1, 1, 0, 2, 0, 0, 2, 2, 1, 4, 0, 1, 4, 1, 1, 0, 4, 3, 1, 1, 2, 3,
		1, 1,
	}},
	{Shape{2,3,4,5}, []int{
		1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,
		1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
		0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
		0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
		1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
		0, 0, 0, 0, 0,
	}},
}
`

const argminCorrect = `var argminCorrect = []struct {
	shape Shape
	data []int
}{
	{Shape{3,4,5,2}, []int{
		0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,
		1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
		0, 1, 1, 0, 1,
	}},
	{Shape{2,4,5,2}, []int{
		2, 1, 0, 0, 1, 2, 1, 2, 1, 2, 1, 0, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 2,
		0, 0, 1, 2, 0, 0, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 1, 2, 1, 2, 1, 2, 1,
		2, 1, 1, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 2,
		2, 0, 0, 0, 1, 2, 2, 2, 2, 1, 1,
	}},
	{Shape{2,3,5,2}, []int{
		0, 1, 0, 2, 2, 1, 3, 2, 3, 2, 1, 0, 3, 3, 0, 1, 0, 3, 0, 2, 0, 1, 0,
		1, 3, 0, 2, 1, 0, 0, 3, 1, 3, 1, 2, 2, 1, 2, 0, 1, 3, 0, 1, 0, 1, 0,
		2, 1, 0, 3, 0, 2, 0, 0, 0, 1, 0, 1, 1, 1,
	}},
	{Shape{2,3,4,2}, []int{
		1, 0, 0, 0, 2, 3, 4, 0, 3, 0, 3, 0, 4, 4, 3, 1, 0, 2, 3, 0, 3, 0, 0,
		2, 4, 4, 3, 4, 2, 3, 0, 0, 4, 0, 1, 3, 3, 2, 0, 4, 2, 1, 4, 2, 4, 0,
		2, 0,
	}},
	{Shape{2,3,4,5}, []int{
		0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1,
		1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
		1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
		1, 1, 1, 0, 1,
	}},
}
`

type ArgMethodTest struct {
	Kind       reflect.Kind
	ArgMethod  string
	ArgAllAxes int
}

const testArgMethodsRaw = `func TestDense_{{title .ArgMethod}}_{{short .Kind}}(t *testing.T){
	assert := assert.New(t)
	var T, {{.ArgMethod}} *Dense
	var err error
	T = basicDense{{short .Kind}}.Clone().(*Dense)
	for i:= 0; i < T.Dims(); i++ {
		if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True({{.ArgMethod}}Correct[i].shape.Eq({{.ArgMethod}}.Shape()), "{{title .ArgMethod}}(%d) error. Want shape %v. Got %v", i, {{.ArgMethod}}Correct[i].shape)
		assert.Equal({{.ArgMethod}}Correct[i].data, {{.ArgMethod}}.Data(), "{{title .ArgMethod}}(%d) error. ", i)
	}
	// test all axes
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Error(err)
		return
	}
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{.ArgAllAxes}}, {{.ArgMethod}}.ScalarValue())

	{{if hasPrefix .Kind.String "float" -}}
	// test with NaN
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,2,{{mathPkg .Kind}}NaN(), 4}))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with NaN: %v", err)
	}
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal(2, {{.ArgMethod}}.ScalarValue(), "NaN test")

	// test with Mask and Nan
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,{{if eq .ArgMethod "argmax"}}9{{else}}-9{{end}},{{mathPkg .Kind}}NaN(), 4}, []bool{false,true,true,false}))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with NaN: %v", err)
	}		
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{if eq .ArgMethod "argmin"}}0{{else}}3{{end}}, {{.ArgMethod}}.ScalarValue(), "Masked NaN test")

	// test with +Inf
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,2,{{mathPkg .Kind}}Inf(1),4}))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with +Inf: %v", err)
	}
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{if eq .ArgMethod "argmax"}}2{{else}}0{{end}}, {{.ArgMethod}}.ScalarValue(), "+Inf test")

   // test with Mask and +Inf
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,{{if eq .ArgMethod "argmax"}}9{{else}}-9{{end}},{{mathPkg .Kind}}Inf(1), 4}, []bool{false,true,true,false}))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with NaN: %v", err)
	}		
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{if eq .ArgMethod "argmin"}}0{{else}}3{{end}}, {{.ArgMethod}}.ScalarValue(), "Masked NaN test")
    
	// test with -Inf
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,2,{{mathPkg .Kind}}Inf(-1),4 }))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with -Inf: %v", err)
	}
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{if eq .ArgMethod "argmin"}}2{{else}}3{{end}}, {{.ArgMethod}}.ScalarValue(), "+Inf test")

	// test with Mask and -Inf
	T = New(WithShape(4), WithBacking([]{{asType .Kind}}{1,{{if eq .ArgMethod "argmax"}}9{{else}}-9{{end}},{{mathPkg .Kind}}Inf(-1), 4}, []bool{false,true,true,false}))
	if {{.ArgMethod}}, err = T.{{title .ArgMethod}}(AllAxes); err != nil {
		t.Errorf("Failed test with NaN: %v", err)
	}		
	assert.True({{.ArgMethod}}.IsScalar())
	assert.Equal({{if eq .ArgMethod "argmin"}}0{{else}}3{{end}}, {{.ArgMethod}}.ScalarValue(), "Masked -Inf test")

	{{end -}}

	// idiotsville
	_, err = T.{{title .ArgMethod}}(10000)
	assert.NotNil(err)

}
`

var (
	argMethodsData *template.Template
	testArgMethods *template.Template
)

func init() {
	argMethodsData = template.Must(template.New("argmethodsData").Funcs(funcs).Parse(argMethodsDataRaw))
	testArgMethods = template.Must(template.New("testArgMethod").Funcs(funcs).Parse(testArgMethodsRaw))
}

func generateArgmethodsTests(f io.Writer, generic Kinds) {
	fmt.Fprintf(f, "/* Test data */\n\n")
	for _, k := range generic.Kinds {
		if isNumber(k) && isOrd(k) {
			op := ArgMethodTestData{k, data}
			argMethodsData.Execute(f, op)
		}
	}
	fmt.Fprintf(f, "\n%s\n%s\n", argmaxCorrect, argminCorrect)
	for _, k := range generic.Kinds {
		if isNumber(k) && isOrd(k) {
			op := ArgMethodTest{k, "argmax", 7}
			testArgMethods.Execute(f, op)
			op = ArgMethodTest{k, "argmin", 11}
			testArgMethods.Execute(f, op)
		}
	}
}
