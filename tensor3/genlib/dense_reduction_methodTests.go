package main

import (
	"io"
	"text/template"
)

const testDenseSumRaw = `var sumTests = []struct {
	name string
	of Dtype
	shape Shape
	along []int

	correctShape Shape
	correct interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{"common case: T.Sum() for {{.}}", {{asType . | title}}, Shape{2,3}, []int{}, ScalarShape(), {{asType .}}(15)},
	{"A.Sum(0) for {{.}}", {{asType . | title}}, Shape{2,3}, []int{0}, Shape{3}, []{{asType .}}{3, 5, 7}},
	{"A.Sum(1) for {{.}}", {{asType . | title}}, Shape{2,3},[]int{1}, Shape{2}, []{{asType .}}{3, 12}},
	{"A.Sum(0,1) for {{.}}", {{asType . | title}}, Shape{2,3},[]int{0, 1}, ScalarShape(), {{asType .}}(15)},
	{"A.Sum(1,0) for {{.}}", {{asType . | title}},  Shape{2,3},[]int{1, 0}, ScalarShape(), {{asType .}}(15)},
	{"3T.Sum(1,2) for {{.}}", {{asType . | title}}, Shape{2,3,4}, []int{1,2}, Shape{2}, []{{asType .}}{66, {{if eq .String "int8"}}-46{{else}}210{{end}} }},
	{{end -}}
	{{end -}}
}
func TestDense_Sum(t *testing.T){
	assert := assert.New(t)
	var T, T2 *Dense
	var err error

	for _, sts := range sumTests {
		T = New(WithShape(sts.shape...), WithBacking(Range(sts.of, 0, sts.shape.TotalSize())))
		if T2, err = T.Sum(sts.along ...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()))
		assert.Equal(sts.correct, T2.Data())
	}

	// idiots
	_,err =T.Sum(1000)
	assert.NotNil(err)
}
`

var (
	testDenseSum *template.Template
)

func init() {
	testDenseSum = template.Must(template.New("testDenseSum").Funcs(funcs).Parse(testDenseSumRaw))
}

func generateDenseReductionMethodsTests(f io.Writer, generic *ManyKinds) {
	testDenseSum.Execute(f, generic)
}
