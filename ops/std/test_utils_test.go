package stdops

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
)

func typecheck(op ops.Op, a, b values.Value) (retType hm.Type, err error) {
	childrenTypes := hm.BorrowTypes(3)
	defer hm.ReturnTypes(childrenTypes)
	childrenTypes = childrenTypes[:0]

	childrenTypes = append(childrenTypes, datatypes.TypeOf(a), datatypes.TypeOf(b))

	childrenTypes = append(childrenTypes, hm.TypeVariable('z'))
	return types.Infer(op.Type(), childrenTypes...)
}

func shapecheck(op ops.Op, a, b values.Value) (retVal shapes.Shape, err error) {
	s := op.ShapeExpr()
	if s, err = shapes.InferApp(s, a.Shape()); err != nil {
		return nil, errors.Wrapf(err, "Unable to infer %v on a. Last inferred shape: %v", op, s)
	}
	if s, err = shapes.InferApp(s, b.Shape()); err != nil {
		return nil, errors.Wrapf(err, "Unable to infer %v on b. Last inferred shape: %v", op, s)
	}
	return shapes.ToShape(s)
}
