package tensorf64

// unsafe only
func (t *Tensor) VAdd(other interface{}) {
	of, ofok := other.(float64)
	ot, otok := other.(*Tensor)

	iter := newIterator(t)
	switch {
	case ofok:
		for i, err := iter.next(); err == nil; i, err = iter.next() {
			t.data[i] += of
		}
	case otok:
		oter := newIterator(ot)
		var err error

		var i, j int
		for {
			i, err = iter.next()
			if err != nil {
				break
			}
			j, err = oter.next()
			if err != nil {
				break
			}

			t.data[i] += ot.data[j]
		}
	}
}
