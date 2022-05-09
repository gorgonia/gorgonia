package stdops

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"log"
)

func typecheck(op ops.Op, vs ...values.Value) (retType hm.Type, err error) {
	childrenTypes := hm.BorrowTypes(3)
	defer hm.ReturnTypes(childrenTypes)
	childrenTypes = childrenTypes[:0]

	ts := make([]hm.Type, len(vs))
	for i := range vs {
		ts[i] = datatypes.TypeOf(vs[i])
	}
	childrenTypes = append(childrenTypes, ts...)

	childrenTypes = append(childrenTypes, hm.TypeVariable('z'))
	return types.Infer(op.Type(), childrenTypes...)
}

func shapecheck(op ops.Op, vs ...values.Value) (retVal shapes.Shape, err error) {
	s := op.ShapeExpr()
	for i, v := range vs {
		if s, err = shapes.InferApp(s, v.Shape()); err != nil {
			return nil, errors.Wrapf(err, "Unable to infer %v on %dth value. Last inferred shape: %v", op, i, s)
		}
	}
	return shapes.ToShape(s)
}
