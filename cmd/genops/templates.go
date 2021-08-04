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

var (
	binopTmpl      *template.Template
	binSymDiffTmpl *template.Template
)

func init() {
	binopTmpl = template.Must(template.New("binop").Funcs(funcmap).Parse(binopRaw))
	binSymDiffTmpl = template.Must(template.New("binsymdiff").Funcs(funcmap).Parse(binSymDiffRaw))
}
