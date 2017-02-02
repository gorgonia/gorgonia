package main

import (
	"io"
	"text/template"
)

const onesTestsRaw = `var onesTests = []struct {
	of Dtype
	shape Shape
	correct interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ {{asType . | title | strip}},  ScalarShape(), {{asType .}}(1)},
	{ {{asType . | title | strip}},  Shape{2,2}, []{{asType .}}{1,1,1,1}},
	{{end -}}
	{{end -}}
	{Bool, ScalarShape(), true},
	{Bool, Shape{2,2}, []bool{true, true, true, true}},
}

func TestOnes(t *testing.T){
	assert := assert.New(t)
	for _, ot := range onesTests{
		T := Ones(ot.of, ot.shape...)
		assert.True(ot.shape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())
	}
}
`

const eyeTestsRaw = `// yes, it's a pun on eye tests, stop asking and go see your optometrist
var eyeTests = []struct{
	E Dtype
	R, C, K int


	correct interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{ {{asType . | title | strip}}, 4,4, 0, []{{asType .}}{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
	{ {{asType . | title | strip}}, 4,4, 1, []{{asType .}}{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 2, []{{asType .}}{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 3, []{{asType .}}{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, 4, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -1, []{{asType .}}{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
	{ {{asType . | title | strip}}, 4,4, -2, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -3, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,4, -4, []{{asType .}}{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{ {{asType . | title | strip}}, 4,5, 0, []{{asType .}}{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}},
	{ {{asType . | title | strip}}, 4,5, 1, []{{asType .}}{0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1}},
	{ {{asType . | title | strip}}, 4,5, -1, []{{asType .}}{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0}},
	{{end -}}
	{{end -}}
}

func TestI(t *testing.T){
	assert := assert.New(t)
	var T Tensor

	for i, it := range eyeTests {
		T = I(it.E, it.R, it.C, it.K)
		assert.True(Shape{it.R, it.C}.Eq(T.Shape()))
		assert.Equal(it.correct, T.Data(), "Test %d-R: %d, C: %d K: %d", i, it.R, it.C, it.K)
	}

}
`

var (
	onesTests *template.Template
	eyeTests  *template.Template
)

func init() {
	onesTests = template.Must(template.New("onesTest").Funcs(funcs).Parse(onesTestsRaw))
	eyeTests = template.Must(template.New("eyeTest").Funcs(funcs).Parse(eyeTestsRaw))
}

func generateDenseTests(f io.Writer, generic *ManyKinds) {
	onesTests.Execute(f, generic)
	eyeTests.Execute(f, generic)
}
