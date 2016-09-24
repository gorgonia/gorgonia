package tensorf32

import "github.com/chewxy/gorgonia/tensor/types"

// unsafe only
func (t *Tensor) VAdd(other interface{}) {
	of, ofok := other.(float32)
	ot, otok := other.(*Tensor)

	iter := types.NewFlatIterator(t.AP)
	switch {
	case ofok:
		for i, err := iter.Next(); err == nil; i, err = iter.Next() {
			t.data[i] += of
		}
	case otok:
		oter := types.NewFlatIterator(ot.AP)
		var err error

		var i, j int
		for {
			i, err = iter.Next()
			if err != nil {
				break
			}
			j, err = oter.Next()
			if err != nil {
				break
			}

			t.data[i] += ot.data[j]
		}
	}
}
