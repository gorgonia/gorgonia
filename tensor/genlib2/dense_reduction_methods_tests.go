package main

import (
	"fmt"
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

const testDenseMaxRaw = `var maxTests = []struct {
	name  string
	of Dtype
	shape Shape
	along []int

	correctShape Shape
	correct  interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{{if isOrd . -}}
	{"common case: T.Max() for {{.}}", {{asType . | title}}, Shape{2,3}, []int{}, ScalarShape(), {{asType .}}(5)},
	{"A.Max(0)", {{asType . | title}}, Shape{2,3},[]int{0}, Shape{3}, []{{asType . }}{3, 4, 5}},
	{"A.Max(1)", {{asType . | title}}, Shape{2,3},[]int{1}, Shape{2}, []{{asType . }}{2,5}},
	{"A.Max(0,1)", {{asType . | title}}, Shape{2,3},[]int{0, 1}, ScalarShape(), {{asType .}}(5)},
	{"A.Max(1,0)", {{asType . | title}}, Shape{2,3},[]int{1, 0}, ScalarShape(), {{asType .}}(5)},
	{"3T.Max(1,2)", {{asType . | title}}, Shape{2,3,4}, []int{1,2}, Shape{2}, []{{asType .}}{11, 23} },
	{{end -}}
	{{end -}}
	{{end -}}
}

func TestDense_Max(t *testing.T){
	assert := assert.New(t)
	var T, T2 *Dense
	var err error

	for _, mts := range maxTests {
		T = New(WithShape(mts.shape...), WithBacking(Range(mts.of, 0, mts.shape.TotalSize())))
		if T2, err = T.Max(mts.along...); err != nil{
			t.Error(err)
			continue
		}
		assert.True(mts.correctShape.Eq(T2.Shape()))
		assert.Equal(mts.correct, T2.Data())
	}
	/* IDIOT TESTING TIME */
	_, err = T.Max(1000)
	assert.NotNil(err)
}
`

const testDenseMinRaw = `var minTests = []struct {
	name  string
	of Dtype
	shape Shape
	along []int

	correctShape Shape
	correct  interface{}
}{
	{{range .Kinds -}}
	{{if isNumber . -}}
	{{if isOrd . -}}
	{"common case: T.Min() for {{.}}", {{asType .|title}}, Shape{2,3}, []int{}, ScalarShape(), {{asType .}}(0)},
	{"A.Min(0)", {{asType .|title}}, Shape{2,3}, []int{0}, Shape{3}, []{{asType .}}{0, 1, 2}},
	{"A.Min(1)", {{asType .|title}}, Shape{2,3}, []int{1}, Shape{2}, []{{asType .}}{0, 3}},
	{"A.Min(0,1)", {{asType .|title}}, Shape{2,3}, []int{0, 1}, ScalarShape(), {{asType .}}(0)},
	{"A.Min(1,0)", {{asType .|title}}, Shape{2,3}, []int{1, 0}, ScalarShape(), {{asType .}}(0)},
	{"3T.Min(1,2)", {{asType . | title}}, Shape{2,3,4}, []int{1,2}, Shape{2}, []{{asType .}}{0,12} },
	{{end -}}
	{{end -}}
	{{end -}}
}

func TestDense_Min(t *testing.T){
	assert := assert.New(t)
	var T, T2 *Dense
	var err error

	for _, mts := range minTests {
		T = New(WithShape(mts.shape...), WithBacking(Range(mts.of, 0, mts.shape.TotalSize())))
		if T2, err = T.Min(mts.along...); err != nil{
			t.Error(err)
			continue
		}
		assert.True(mts.correctShape.Eq(T2.Shape()))
		assert.Equal(mts.correct, T2.Data())
	}

	/* IDIOT TESTING TIME */
	_, err = T.Min(1000)
	assert.NotNil(err)
}
`

var (
	testDenseSum *template.Template
	testDenseMax *template.Template
	testDenseMin *template.Template
)

func init() {
	testDenseSum = template.Must(template.New("testDenseSum").Funcs(funcs).Parse(testDenseSumRaw))
	testDenseMax = template.Must(template.New("testDenseMax").Funcs(funcs).Parse(testDenseMaxRaw))
	testDenseMin = template.Must(template.New("testDenseMin").Funcs(funcs).Parse(testDenseMinRaw))
}

func generateDenseReductionMethodsTests(f io.Writer, generic Kinds) {
	testDenseSum.Execute(f, generic)
	fmt.Fprint(f, "\n")
	testDenseMax.Execute(f, generic)
	fmt.Fprint(f, "\n")
	testDenseMin.Execute(f, generic)
}
