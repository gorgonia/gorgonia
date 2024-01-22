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
		return nil, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.{{.Method}}(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
{{- end -}}
{{- define "PreallocDo" -}}
if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
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
		return nil, err
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
		return nil, err
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
func (op {{.Name}}VS) String() string { return "{{.Symbol}}·" }

{{ template "Type()VS" . }}



// {{.Name}}SV is a scalar-tensor {{.CommentOp}}.
{{- template "TypeDefSV" . -}}

// String implements fmt.Stringer.
func (op {{.Name}}SV) String() string { return "·{{.Symbol}}" }

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
	op := {{$VV}}{ {{if .IsCmpRetTrue}}{{.Name}}Op{retSame: true}, binopVV{} {{end}} }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do (using tensor-tensor) */

	// set up
	var a, b, c values.Value
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
	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	b = tensor.New(tensor.WithShape(), tensor.Of(tensor.Float64))
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VV{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf(a), datatypes.TypeOf(b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VV{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test_{{$VS}}{{if .IsCmpRetTrue}}_RetSame{{end}}(t *testing.T) {
	op := {{$VS}}{ {{if .IsCmpRetTrue}}{{.Name}}Op{retSame: true}, binopVS{} {{end}} }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c values.Value
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

	b = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	// we won't type check because the type system is not a dependent type system, thus
	// {{.Name}}VS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}VS{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test_{{$SV}}{{if .IsCmpRetTrue}}_RetSame{{end}}(t *testing.T) {
	op := {{$SV}}{ {{if .IsCmpRetTrue}}{{.Name}}Op{retSame: true}, binopSV{} {{end}}  }
	// basic test
	assert.Equal(t, 2, op.Arity())

	/* Do */

	// set up
	var a, b, c values.Value
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

	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
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
func (op {{.Name}}Op[DT,T]) Do(ctx context.Context, vs ...values.Value)(retVal values.Value, err error){
if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.{{.Method}}(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs {{.CommentOp}} but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op {{.Name}}Op[DT,T]) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.{{.Method}}(a, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}


{{ if  .IsDiff }}
// DiffWRT returns {true} for {{.Name}}
func (op {{.Name}}Op[DT,T]) DiffWRT(inputs int) []bool { return onetrue }
{{- end -}}

`

const unopTestRaw = `func Test_{{.Name}}(t *testing.T){
	op := {{.Name}}Op{}
	// basic test
	assert.Equal(t, 1, op.Arity())

	// Do
	var a, b values.Value
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking({{.Input}}))

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
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-100, -100, -100, -100, -100, -100}))
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
{{- $vv = (printf "%vVV{%vOp{retSame: retSame}, binopVV{}}" .Name .Name ) -}}
{{- $vs = (printf "%vVS{%vOp{retSame: retSame}, binopVS{}}" .Name .Name ) -}}
{{- $sv = (printf "%vSV{%vOp{retSame: retSame}, binopSV{}}" .Name .Name ) -}}
{{- else -}}
{{- $cmp = ""}}
{{- $vv = (printf "%vVV{}" .Name ) -}}
{{- $vs = (printf "%vVS{}" .Name ) -}}
{{- $sv = (printf "%vSV{}" .Name ) -}}
{{- end -}}

// {{.Name | title}} creates an ops.Op that is correct to the shape of the given operands.
func {{.Name | title}}(a, b ops.Operand {{$cmp}}) ops.PreallocOp {
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

	var op, expected ops.Op

	// test vv
	a := tensor.New(tensor.WithShape(2,3), tensor.Of(tensor.Float64))
	b := tensor.New(tensor.WithShape(2,3), tensor.Of(tensor.Float64))
	op = {{.Name | title}}(a, b {{$retSameFalse}})
	expected = {{.Name}}VV{ {{.Name}}Op{ {{$cmpfalse}} }, binopVV{} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test vv but retSame = true
	op = {{.Name | title}}(a, b {{$retSameTrue}})
	expected = {{.Name}}VV{ {{.Name}}Op{ {{$cmptrue}} }, binopVV{} }
	assert.Equal(op, expected)
{{ end }}

	// test vs
	b = tensor.New(tensor.WithShape(), tensor.Of(tensor.Float64))
	op = {{.Name | title}}(a, b {{$retSameFalse}})
	expected = {{.Name}}VS{ {{.Name}}Op{ {{$cmpfalse}} }, binopVS{} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test vs but retSame = true
	op = {{.Name | title}}(a, b {{$retSameTrue}})
	expected = {{.Name}}VS{ {{.Name}}Op{ {{$cmptrue}} }, binopVS{} }
	assert.Equal(op, expected)
{{ end }}


	// test sv
	op = {{.Name | title}}(b, a {{$retSameFalse}})
	expected = {{.Name}}SV{ {{.Name}}Op{ {{$cmpfalse}} }, binopSV{} }
	assert.Equal(op, expected)

{{ if .IsCmp }}
	// test sv but retSame = true
	op = {{.Name | title}}(b, a  {{$retSameTrue}})
	expected = {{.Name}}SV{ {{.Name}}Op{ {{$cmptrue}} }, binopSV{} }
	assert.Equal(op, expected)
{{ end }}


	// test ss
	a = tensor.New(tensor.WithShape(), tensor.Of(tensor.Float64))
	op = {{.Name | title}}(a, b {{$retSameFalse}})
	expected = {{.Name}}VV{ {{.Name}}Op{ {{$cmpfalse}} }, binopVV{} }
	assert.Equal(op, expected)


{{ if .IsCmp }}
	// test ss but retSame = true
	op = {{.Name | title}}(a, b {{$retSameTrue}})
	expected = {{.Name}}VV{ {{.Name}}Op{ {{$cmptrue}} }, binopVV{} }
	assert.Equal(op, expected)
{{ end }}

}

`

const doDiffTmplRaw = `{{ if .IsDiff }}
// DoDiff is the method that allows automatic differentiation of` + " `{{ .Name }}` " + `g.
func (op {{ .Name }}Op) DoDiff(ctx context.Context, inputs []Tensor, output Tensor) error {
	adv := exprgraph.T2T(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2T(inputs[1]).(*dual.Dual)
	cdv := exprgraph.T2T(output).(*dual.Dual)

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	_, _, _ = cdv, advd, bdvd
	panic("Not implemented")
}
{{ end }}

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
}
