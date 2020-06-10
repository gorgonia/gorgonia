package exprgraph

import "gorgonia.org/tensor"

func WithName(name string) tensor.ConsOpt {
	return func(t tensor.Tensor) {
		en := t.Engine()
		if e, ok := en.(*Graph); ok {
			id := e.idOrInsert(t)
			e.nodes[id].name = name
		}

	}
}

func inGraph() tensor.ConsOpt {
	return func(t tensor.Tensor) {
		en := t.Engine()
		if e, ok := en.(*Graph); ok {
			e.idOrInsert(t)
		}
	}
}

func WithChildren(a []tensor.Tensor) tensor.ConsOpt {
	return func(t tensor.Tensor) {
		en := t.Engine()
		if e, ok := en.(*Graph); ok {
			id := e.idOrInsert(t)
			ids := make([]NodeID, len(a))
			for i, child := range a {
				ids[i] = NodeID(e.ID(child))
				if ids[i] == -1 {
					panic("Tensor not in graph")
				}
			}
			e.AddChildren(NodeID(id), ids)
		}
	}
}
