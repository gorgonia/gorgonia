package main

import (
	"text/template"
)

const arithMetaRaw = `
{{define "TypeDefRaw"}}
type {{.Name }}Op[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op {{.Name}}Op[DT,T]) String() string { return "{{.Symbol}}" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}Op[DT,T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	{{- template "Do" . -}}
}

// PreallocDo performs {{.CommentOp}} but with a preallocated return value.
// PreallocDo allows {{.Name}} to implement ops.PreallocOp.
func (op {{.Name}}Op[DT,T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	{{- template "PreallocDo" . -}}
}


{{- if not .IsDiff -}}
// DiffWRT returns {false, false} for {{.Name}}
func (op {{.Name}}Op[DT,T]) DiffWRT(inputs int) []bool { return twofalses }
{{- end -}}
{{end}}

{{define "TypeDefVV"}}
type {{.Name}}VV[DT any, T values.Value[DT]] struct { {{.Name }}Op[DT,T] ; binopVV }
{{end}}

{{define "TypeDefVS"}}
type {{.Name}}VS[DT any, T values.Value[DT]] struct { {{.Name}}Op[DT,T] ; binopVS }
{{end}}

{{define "TypeDefSV"}}
type {{.Name}}SV[DT any, T values.Value[DT]] struct { {{.Name}}Op[DT,T] ; binopSV }
{{end}}

{{- define "Do" -}}
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())

	e := getEngine(a, b)
	var {{.InterfaceName | lower}} tensor.{{.InterfaceName}}[DT, T]
	var ok bool
	if {{.InterfaceName | lower}}, ok = e.(tensor.{{.InterfaceName}}[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, {{.InterfaceName | lower}}, errors.ThisFn())
	}

	ashp := a.Shape()
	bshp := b.Shape()
	expShape := getLargestShape(ashp, bshp)
	var fo tensor.Option
	if retVal, fo, err = handleFuncOpts[DT](e, a, expShape); err != nil {
		return retVal, err
	}

	switch {
	case ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, b, retVal, fo.Incr)
	case ashp.IsScalarEquiv() && !bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}Scalar(ctx2, b, a.Data()[0], retVal, true, fo.Incr)
	case !ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}Scalar(ctx2, a, b.Data()[0], retVal, false, fo.Incr)
	default:
		err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, b, retVal, fo.Incr)
	}

	task.End()
	return retVal, err
{{- end -}}
{{- define "PreallocDo" -}}
if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())

	e := getEngine(a, b)
	var {{.InterfaceName | lower}} tensor.{{.InterfaceName}}[DT, T]
	var ok bool
	if {{.InterfaceName | lower}}, ok = e.(tensor.{{.InterfaceName}}[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, {{.InterfaceName | lower}}, errors.ThisFn())
	}

	ashp := a.Shape()
	bshp := b.Shape()
	expShape := getLargestShape(ashp, bshp)
	var fo tensor.Option
	if retVal, fo, err = handleFuncOpts[DT](e, a, expShape, tensor.WithReuse(prealloc)); err != nil {
		return retVal, err
	}

	switch {
	case ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, b, retVal, fo.Incr)
	case ashp.IsScalarEquiv() && !bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}Scalar(ctx2, b, a.Data()[0], retVal, true, fo.Incr)
	case !ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = {{.InterfaceName | lower}}.{{.Method}}Scalar(ctx2, a, b.Data()[0], retVal, false, fo.Incr)
	default:
		err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, b, retVal, fo.Incr)
	}

	task.End()
	return retVal, err
{{- end -}}

{{/* we don't need to generate Type() for arithmetic Ops */}}
{{define "Type()VV"}}{{end}}
{{define "Type()VS"}}{{end}}
{{define "Type()SV"}}{{end}}
`

const cmpMetaRaw = `

{{define "TypeDefRaw"}}
type {{.Name}}Op[DT any, T values.Value[DT]] struct{ binop; retSame bool }

// String implements fmt.Stringer.
func (op {{.Name}}Op[DT,T]) String() string { return "{{.Symbol}}" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}Op[DT,T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	{{- template "Do" . -}}
}

// PreallocDo performs {{.CommentOp}} but with a preallocated return value.
// PreallocDo allows {{.Name}} to implement ops.PreallocOp.
func (op {{.Name}}Op[DT,T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	{{- template "PreallocDo" . -}}
}

{{- if not .IsDiff -}}
// DiffWRT returns {false, false} for {{.Name}}
func (op {{.Name}}Op[DT,T]) DiffWRT(inputs int) []bool { return twofalses }
{{- end -}}
{{end}}


{{define "TypeDefVV"}}
type {{.Name}}VV[DT any, T values.Value[DT]] struct { {{.Name}}Op[DT,T]; binopVV  }
{{end}}

{{define "TypeDefVS"}}
type {{.Name}}VS[DT any, T values.Value[DT]] struct { {{.Name}}Op[DT,T]; binopVS }
{{end}}

{{define "TypeDefSV"}}
type {{.Name}}SV[DT any, T values.Value[DT]] struct { {{.Name}}Op[DT,T]; binopSV }
{{end}}

{{- define "Do" -}}
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	if op.retSame{
		retVal, err = tensor.{{.Method}}(a, b, tensor.WithContext(ctx2), tensor.AsSameType())
	} else {
		retVal, err = tensor.{{.Method}}(a, b, tensor.WithContext(ctx2))
	}
	task.End()
	return retVal, err
{{- end -}}
{{- define "PreallocDo" -}}
if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	if op.retSame {
	retVal, err = tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2), tensor.AsSameType())
	} else {
	retVal, err = tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	}
	task.End()
	return retVal, err
{{- end -}}

{{define "Type()VV"}}
// Type returns the type: (·) : a → a → a or (·) :  a → a → b
func (op {{.Name}}VV[DT,T]) Type() hm.Type{
	a := hm.TypeVariable('a') // (T U) or U
	if op.retSame{
		return types.NewFunc(a, a, a)
	}
	b := types.MakeDependent(a, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a,a,b)
}
{{end}}
{{define "Type()VS"}}
// Type returns the type: (·) : a → b → a or (·) :  a → b → c
func (op {{.Name}}VS) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	b := hm.TypeVariable('b') // U
	if op.retSame{
		return types.NewFunc(a, b, a)
	}
	c := types.MakeDependent(a, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a,b,c)
}
{{end}}
{{define "Type()SV"}}
// Type returns the type: (·) : a → b → b or (·) :  a → b → c
func (op {{.Name}}SV[DT,T]) Type() hm.Type {
	a := hm.TypeVariable('a') // U
	b := hm.TypeVariable('b') // (T U) or U
	if op.retSame{
		return types.NewFunc(a, b, b)
	}
	c := types.MakeDependent(b, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a,b,c)
}
{{end}}
`

const binOpRaw = `// {{.Name}}Op is the base op for {{.CommentOp}}.
{{- template "TypeDefRaw" . }}

// {{.Name}}VV is a tensor-tensor {{.CommentOp}}.
{{- template "TypeDefVV" . -}}



{{ template "Type()VV" . }}


// {{.Name}}VS is a tensor-scalar {{.CommentOp}}.
{{- template "TypeDefVS" . -}}

// String implements fmt.Stringer.
func (op {{.Name}}VS[DT,T]) String() string { return "{{.Symbol}}·" }

{{ template "Type()VS" . }}



// {{.Name}}SV is a scalar-tensor {{.CommentOp}}.
{{- template "TypeDefSV" . -}}

// String implements fmt.Stringer.
func (op {{.Name}}SV[DT,T]) String() string { return "·{{.Symbol}}" }

{{ template "Type()SV" . }}

`

const binSymDiffRaw = `{{ if .IsDiff }}
// SymDiff performs the symbolic differentiation of {{.Name}}.
func (op {{.Name}}Op[DT,T])SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error){ panic("not implemented" )}
{{ end }}

`

const arithOpTestRaw = `{{ define "varExpected" }}
	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error
{{end}}
{{define "typeshapecheck"}}
	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.}}{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.}}{} to pass shape checking. Err: %v", err)
	}
{{end}}
{{ define "op.Do"}}
	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected {{.}}{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
{{end}}
{{ define "op.PreallocDo" }}
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected {{.}}{}'s Prealloc to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
{{ end }}

{{- $VV := ( printf "%vVV" .Name ) -}}
{{- $VS := ( printf "%vVS" .Name ) -}}
{{- $SV := ( printf "%vSV" .Name ) -}}
func Test_{{$VV}}{{if .IsCmpRetTrue}}_RetSame{{end}}(t *testing.T) {
	op := {{$VV}}[float64, *dense.Dense[float64]]{ {{if .IsCmpRetTrue}}{{.Name}}Op[float64, *dense.Dense[float64]]{retSame: true}, binopVV{} {{end}} }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do (using tensor-tensor) */

	// set up
	var a, b, c *dense.Dense[float64]
	{{- template "varExpected" }}
	a = {{.AVV}}
	b = {{.BVV}}

	// type and shape checks
	{{-  template "typeshapecheck" $VV }}

	// actually doing and testing
	{{- template "op.Do" $VV -}}
	correct := {{.Correct}}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo (using scalar-scalar to make sure things don't go wrong) */

	// set up
	a = {{.AVV2}}
	b = {{.BVV2}}
	c = {{.CVV}}

	// type and shape checks
	{{- template "typeshapecheck" $VV }}

	// actually PreallocDo-ing and testing
	{{- template "op.PreallocDo" $VV -}}
	correctScalar := {{.CorrectScalar}}
	assert.Equal(t, correctScalar, c.Data())


	// bad cases: fails  typecheck and shapecheck
	a = dense.New[float64](tensor.WithShape(2, 3))
	b = dense.New[float64](tensor.WithShape())
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VV{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf(a), datatypes.TypeOf(b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VV{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test_{{$VS}}{{if .IsCmpRetTrue}}_RetSame{{end}}(t *testing.T) {
	op := {{$VS}}[float64, *dense.Dense[float64]]{ {{if .IsCmpRetTrue}}{{.Name}}Op[float64, *dense.Dense[float64]]{retSame: true}, binopVS{} {{end}} }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	{{- template "varExpected" }}
	a = {{.AVS}}
	b = {{.BVS}}

	// type and shape checks
	{{- template "typeshapecheck" $VS }}

	// actually doing and test
	{{- template "op.Do" $VS -}}
	correct := {{.CorrectVS}}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = {{.CVS}}

	// actually PreallocDo-ing and checking
	{{- template "op.PreallocDo" $VS -}}
	assert.Equal(t, correct, c.Data())

	/* bad cases: {{$VS}}{} on tensor-tensor */

	b = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// {{.Name}}VS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VS{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_{{$SV}}{{if .IsCmpRetTrue}}_RetSame{{end}}(t *testing.T) {
	op := {{$SV}}[float64, *dense.Dense[float64]]{ {{if .IsCmpRetTrue}}{{.Name}}Op[float64, *dense.Dense[float64]]{retSame: true}, binopSV{} {{end}}  }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c *dense.Dense[float64]
	{{- template "varExpected" }}
	a = {{.ASV}}
	b = {{.BSV}}


	// type and shape checks
	{{- template "typeshapecheck" $SV }}

	// actually doing and test
	{{- template "op.Do" $SV -}}
	correct := {{.CorrectSV}}
	assert.Equal(t, correct, c.Data())

	/* PreallocDo */

	// set up - create a new preallocated result
	c = {{.CSV}}

	// actually PreallocDo-ing and checking
	{{- template "op.PreallocDo" $VS -}}
	assert.Equal(t, correct, c.Data())

	/* bad cases: {{.Name}}SV{} on tensor-tensor */

	a = dense.New[float64](tensor.WithShape(2, 3))
	// we won't type check because the type system is not a dependent type system, thus
	// {{.Name}}SV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}SV{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}
`

const unopTmplRaw = `// {{.Name}} is a {{.CommentOp}}.
type {{.Name}}Op[DT any, T values.Value[DT]] struct{unop}

// String implements fmt.Stringer.
func (op {{.Name}}Op[DT,T]) String() string {return "{{.Symbol}}" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}Op[DT,T]) Do(ctx context.Context, vs ...T)(retVal T, err error){
if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := getEngine(a)
	var {{.InterfaceName | lower}} {{.InterfaceName}}[DT,T]
	var ok bool
	if {{.InterfaceName | lower}} = e.({{.InterfaceName}}[DT,T]); !ok{
		return retVal, errors.Errorf(errors.EngineSupport, e, {{.InterfaceName | lower}}, errors.ThisFn())
	}
	if retVal, _, err = handleFuncOpts[DT,T] (e, a, a.Shape()); err !=nil{
		return retVal, errors.Wrapf(err , errors.FailedFuncOpt, errors.ThisFn())
	}
	if err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, retVal); err !=nil{
		return retVal, err
	}
	// retVal, err = tensor.{{.Method}}(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs {{.CommentOp}} but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op {{.Name}}Op[DT,T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := getEngine(a)
	var {{.InterfaceName | lower}} {{.InterfaceName}}[DT,T]
	var ok bool
	if {{.InterfaceName | lower}} = e.({{.InterfaceName}}[DT,T]); !ok{
		return retVal, errors.Errorf(errors.EngineSupport, e, {{.InterfaceName | lower}}, errors.ThisFn())
	}
	// TODO check that prealloc has the same shape as expected reetVal shape
	if err = {{.InterfaceName | lower}}.{{.Method}}(ctx2, a, prealloc); err != nil{
		return retVal, err
	}
	task.End()
	return retVal, err
}


{{ if  .IsDiff }}
// DiffWRT returns {true} for {{.Name}}
func (op {{.Name}}Op[DT,T]) DiffWRT(inputs int) []bool { return onetrue }
{{- end -}}

`

const unopTestRaw = `func Test_{{.Name}}(t *testing.T){
	op := {{.Name}}Op[float64, *dense.Dense[float64]]{}
	// basic test
	assert.Equal(t, 1, op.Arity())

	// Do
	var a, b values.Value
	a = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking({{.Input}}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a); err != nil {
		t.Fatalf("{{.Name}}Op failed to typecheck. Err: %v", err)
	}

	if expectedShape, err = shapecheck(op, a); err != nil {
		t.Fatalf("{{.Name}}Op failed to shapecheck. Err: %v", err)
	}

	if b, err = op.Do(context.Background(), a); err != nil {
		t.Fatalf("Expected {{.Name}}Op{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	correct := {{.Correct}}
	assert.Equal(t, correct, b.Data())

	// PreallocDo
	b = dense.New[float64](tensor.WithShape(2, 3), tensor.WithBacking([]float64{-100, -100, -100, -100, -100, -100}))
	if b, err = op.PreallocDo(context.Background(), b, a); err != nil {
		t.Fatalf("Expected {{.Name}}Op{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(b))
	assert.True(t, expectedShape.Eq(b.Shape()))
	assert.Equal(t, correct, b.Data())
}
`

const binopAPIRaw = `{{- $cmp := "" -}}
{{- $vv := "" -}}
{{- $vs := "" -}}
{{- $sv := "" -}}
{{- if .IsCmp -}}
{{- $cmp = ", retSame bool" -}}
{{- $vv = (printf "%vVV[DT,T]{%vOp[DT,T]{retSame: retSame}, binopVV{}}" .Name .Name ) -}}
{{- $vs = (printf "%vVS[DT,T]{%vOp[DT,T]{retSame: retSame}, binopVS{}}" .Name .Name ) -}}
{{- $sv = (printf "%vSV[DT,T]{%vOp[DT,T]{retSame: retSame}, binopSV{}}" .Name .Name ) -}}
{{- else -}}
{{- $cmp = ""}}
{{- $vv = (printf "%vVV[DT, T]{}" .Name ) -}}
{{- $vs = (printf "%vVS[DT,T]{}" .Name ) -}}
{{- $sv = (printf "%vSV[DT,T]{}" .Name ) -}}
{{- end -}}

// {{.Name | title}} creates an ops.Op that is correct to the shape of the given operands.
func {{.Name | title}}[DT any, T values.Value[DT]](a, b ops.Operand {{$cmp}}) ops.PreallocOp[DT,T] {
	aScalar := a.Shape().IsScalar()
	bScalar := b.Shape().IsScalar()

	switch {
	default:
		fallthrough
	case !aScalar && !bScalar:
		return {{$vv}}
	case !aScalar && bScalar:
		return {{$vs}}
	case aScalar && !bScalar:
		return {{$sv}}
	}
}

`

const binopAPITestRaw = `{{- $retSameFalse := "" -}}
{{- $retSameTrue := "" -}}
{{- $cmptrue := "" -}}
{{- $cmpfalse := "" -}}
{{- if .IsCmp -}}
{{- $retSameFalse = ", false" -}}
{{- $retSameTrue = ", true" -}}
{{- $cmptrue = "retSame: true" -}}
{{- $cmpfalse = "retSame: false" -}}
{{- end -}}

func Test{{.Name | title}}(t *testing.T){
	assert := assert.New(t)

	var op, expected ops.Op[float64, *dense.Dense[float64]]

	// test vv
	a := dense.New[float64](tensor.WithShape(2,3))
	b := dense.New[float64](tensor.WithShape(2,3))
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameFalse}})
	expected = {{.Name}}VV[float64, *dense.Dense[float64]]{ {{if .IsCmp}} {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmpfalse}} }, binopVV{} {{end}} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test vv but retSame = true
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameTrue}})
	expected = {{.Name}}VV[float64, *dense.Dense[float64]]{ {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmptrue}} }, binopVV{} }
	assert.Equal(op, expected)
{{ end }}

	// test vs
	b = dense.New[float64](tensor.WithShape())
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameFalse}})
	expected = {{.Name}}VS[float64, *dense.Dense[float64]]{ {{if .IsCmp}} {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmpfalse}} }, binopVS{} {{end}} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test vs but retSame = true
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameTrue}})
	expected = {{.Name}}VS[float64, *dense.Dense[float64]]{ {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmptrue}} }, binopVS{} }
	assert.Equal(op, expected)
{{ end }}


	// test sv
	op = {{.Name | title}}[float64, *dense.Dense[float64]](b, a {{$retSameFalse}})
	expected = {{.Name}}SV[float64, *dense.Dense[float64]]{ {{if .IsCmp}} {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmpfalse}} }, binopSV{} {{end}} }
	assert.Equal(op, expected)

{{ if .IsCmp }}
	// test sv but retSame = true
	op = {{.Name | title}}[float64, *dense.Dense[float64]](b, a  {{$retSameTrue}})
	expected = {{.Name}}SV[float64, *dense.Dense[float64]]{ {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmptrue}} }, binopSV{} }
	assert.Equal(op, expected)
{{ end }}


	// test ss
	a = dense.New[float64](tensor.WithShape())
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameFalse}})
	expected = {{.Name}}VV[float64, *dense.Dense[float64]]{ {{if .IsCmp}} {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmpfalse}} }, binopVV{} {{end}} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test ss but retSame = true
	op = {{.Name | title}}[float64, *dense.Dense[float64]](a, b {{$retSameTrue}})
	expected = {{.Name}}VV[float64, *dense.Dense[float64]]{ {{.Name}}Op[float64, *dense.Dense[float64]]{ {{$cmptrue}} }, binopVV{} }
	assert.Equal(op, expected)
{{ end }}

}

`

const doDiffTmplRaw = `{{ if .IsDiff }}
// DoDiff is the method that allows automatic differentiation of` + " `{{ .Name }}` " + `g.
func (op {{ .Name }}Op[DT,T]) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	adv := exprgraph.T2B[DT](inputs[0]).(*dual.Dual[DT,T])
	bdv := exprgraph.T2B[DT](inputs[1]).(*dual.Dual[DT,T])
	cdv := exprgraph.T2B[DT](output).(*dual.Dual[DT,T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	_, _, _ = cdv, advd, bdvd
	panic("Not implemented")
}
{{ end }}
`

const unopInterfaceTemplRaw = `
type {{.InterfaceName}}[DT any, T tensor.Basic[DT]] interface{
	{{range .Ops -}}
	{{.Method}}(ctx context.Context, a, retVal T) error
	{{end -}}
}
`

var (
	arithMetaTmpl    *template.Template
	arithOpTmpl      *template.Template
	cmpMetaTmpl      *template.Template
	cmpOpTmpl        *template.Template
	binSymDiffTmpl   *template.Template
	arithOpTestTmpl  *template.Template
	unopTmpl         *template.Template
	unopTestTmpl     *template.Template
	binopAPITmpl     *template.Template
	binopAPITestTmpl *template.Template

	doDiffTmpl *template.Template

	unopInterfaceTempl *template.Template
)

func init() {
	arithMetaTmpl = template.Must(template.New("arith meta-templates").Funcs(funcmap).Parse(arithMetaRaw))
	arithOpTmpl = template.Must(arithMetaTmpl.New("arith").Funcs(funcmap).Parse(binOpRaw))
	cmpMetaTmpl = template.Must(template.New("cmp meta-templates").Funcs(funcmap).Parse(cmpMetaRaw))
	cmpOpTmpl = template.Must(cmpMetaTmpl.New("cmp").Funcs(funcmap).Parse(binOpRaw))
	binSymDiffTmpl = template.Must(template.New("binsymdiff").Funcs(funcmap).Parse(binSymDiffRaw))
	arithOpTestTmpl = template.Must(template.New("binopTest").Funcs(funcmap).Parse(arithOpTestRaw))
	unopTmpl = template.Must(template.New("unary op").Funcs(funcmap).Parse(unopTmplRaw))
	unopTestTmpl = template.Must(template.New("unary op test").Funcs(funcmap).Parse(unopTestRaw))
	binopAPITmpl = template.Must(template.New("api").Funcs(funcmap).Parse(binopAPIRaw))
	binopAPITestTmpl = template.Must(template.New("api test").Funcs(funcmap).Parse(binopAPITestRaw))
	doDiffTmpl = template.Must(template.New("binop DoDiff").Funcs(funcmap).Parse(doDiffTmplRaw))

	unopInterfaceTempl = template.Must(template.New("unop interfae").Funcs(funcmap).Parse(unopInterfaceTemplRaw))
}
