package main

import "text/template"

const binopRaw = `// {{.Name}} is a tensor-tensor {{.CommentOp}}
type {{.Name}} struct{ binop }

// String implements fmt.Stringer.
func (op {{.Name}}) String() string { return "{{.Symbol}}" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op {{.Name}}) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}


// {{.Name}}VS is a tensor-scalar {{.CommentOp}}
type {{.Name}}VS struct { binopVS }

// String implements fmt.Stringer.
func (op {{.Name}}VS) String() string { return "{{.Symbol}}·" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}VS) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op {{.Name}}VS) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}


// {{.Name}}SV is a scalar-tensor {{.CommentOp}}
type {{.Name}}SV struct{ binopSV }

// String implements fmt.Stringer.
func (op {{.Name}}SV) String() string { return "·{{.Symbol}}" }

// Do performs {{.CommentOp}}.
func (op {{.Name}}SV) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op {{.Name}}SV) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.{{.Method}}(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

`

const binSymDiffRaw = `func (op {{.Name}})SymDiff(inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error){ panic("not implemented" )}

func (op {{.Name}}VS)SymDiff(inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error){ panic("not implemented" )}

func (op {{.Name}}SV)SymDiff(inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error){ panic("not implemented" )}

`

const binopTestRaw = `func Test{{.Name}}(t *testing.T) {
	op := {{.Name}}{}
	// basic test
	assert.Equal(t, 2, op.Arity())

	// tensor-tensor / Do()

	var a, b, c values.Value
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected {{.Name}}{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := {{.Correct}}
	assert.Equal(t, correct, c.Data())

	// scalar-scalar / PreallocDo

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	c = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{-1}))

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}{} to pass shape checking. Err: %v", err)
	}

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected {{.Name}}{}'s Prealloc to work. Err: %v", err)
	}
	correctScalar := {{.CorrectScalar}}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.Equal(t, correctScalar, c.Data())
	assert.True(t, expectedShape.Eq(c.Shape()))

	// bad cases: fails  typecheck and shapecheck
	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	b = tensor.New(tensor.WithShape(), tensor.Of(tensor.Float64))
	if expectedType, err = typecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}{} to NOT pass type checking. Got ~(%v %v) =  %v ", datatypes.TypeOf(a), datatypes.TypeOf(b), expectedType)
	}
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}{} to NOT pass shape checking. Got expectedShape = %v", expectedShape)
	}

}

func Test{{.Name}}VS(t *testing.T) {
	op := {{.Name}}VS{}

	// Do
	var a, b, c values.Value
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}VS{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}VS{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected {{.Name}}VS{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := {{.CorrectVS}}
	assert.Equal(t, correct, c.Data())

	// PreallocDo
	c = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	// bad cases: {{.Name}}VS{} on tensor-tensor
	b = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	// we won't type check because the type system is not a dependent type system, thus
	// {{.Name}}VS : (a → b → a) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

func Test{{.Name}}SV(t *testing.T) {
	op := {{.Name}}SV{}

	// Do
	var a, b, c values.Value
	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{100}))
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))

	var expectedType hm.Type
	var expectedShape shapes.Shape
	var err error

	if expectedType, err = typecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}SV{} to pass type checking. Err: %v", err)
	}
	if expectedShape, err = shapecheck(op, a, b); err != nil {
		t.Fatalf("Expected {{.Name}}SV{} to pass shape checking. Err: %v", err)
	}

	if c, err = op.Do(context.Background(), a, b); err != nil {
		t.Fatalf("Expected {{.Name}}SV{} to work correctly. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	correct := {{.CorrectSV}}
	assert.Equal(t, correct, c.Data())

	// PreallocDo
	c = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{-1, -1, -1, -1, -1, -1}))

	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work. Err: %v", err)
	}
	assert.Equal(t, expectedType, datatypes.TypeOf(c))
	assert.True(t, expectedShape.Eq(c.Shape()))
	assert.Equal(t, correct, c.Data())

	// bad cases: {{.Name}}SV{} on tensor-tensor
	a = tensor.New(tensor.WithShape(2, 3), tensor.Of(tensor.Float64))
	// we won't type check because the type system is not a dependent type system, thus
	// {{.Name}}SV : (a → b → b) will always type check without errors
	if expectedShape, err = shapecheck(op, a, b); err == nil {
		t.Fatalf("Expected {{.Name}}{} to NOT pass shape checking. Got %v ~ (%v, %v) = %v", op.ShapeExpr(), a.Shape(), b.Shape(), expectedShape)
	}
}

`

var (
	binopTmpl      *template.Template
	binSymDiffTmpl *template.Template
	binopTestTmpl  *template.Template
)

func init() {
	binopTmpl = template.Must(template.New("binop").Funcs(funcmap).Parse(binopRaw))
	binSymDiffTmpl = template.Must(template.New("binsymdiff").Funcs(funcmap).Parse(binSymDiffRaw))
	binopTestTmpl = template.Must(template.New("binopTest").Funcs(funcmap).Parse(binopTestRaw))
}
