package exprgraph

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/shapes"
)

// Apply creates a new *Node
func (g *Graph) Apply(op ops.Op, newname string, children ...*Node) (*Node, error) {
	s := op.ShapeExpr()
	t := op.Type()

	// type check
	childrenTypes := hm.BorrowTypes(len(children) + 1)
	defer hm.ReturnTypes(childrenTypes)
	childrenTypes = childrenTypes[:0] // reset length to 0
	for _, c := range children {
		childrenTypes = append(childrenTypes, datatypes.TypeOf(c.Tensor))
	}
	childrenTypes = append(childrenTypes, hm.TypeVariable('z'))
	retType, err := types.Infer(t, childrenTypes...)
	if err != nil {
		return nil, errors.Wrapf(err, "Unable to apply %v. Type inference failed", op)
	}
	dt, err := datatypes.DtypeOf(retType)
	if err != nil {
		return nil, errors.Wrapf(err, "Unable to get Dtype of inferred return type %v", retType)
	}

	// shape check
	for _, c := range children {
		if s, err = shapes.InferApp(s, c.Shape()); err != nil {
			return nil, errors.Wrapf(err, "Unable to infer %v on children. Last inferred shape %v", op.ShapeExpr(), s)
		}
	}
	shp, err := shapes.ToShape(s)
	if err != nil {
		return nil, errors.Wrapf(err, "Unable to get Shape of shape expression %v", s)
	}
	T := newHeader(g, dt, shp)
	n, err := cons(g, newname, T)
	if err != nil {
		return nil, errors.Wrap(err, "Unable to construct a new Node")
	}
	n.Op = op

	err = g.AddChildren(n, children...)
	return n, err
}
