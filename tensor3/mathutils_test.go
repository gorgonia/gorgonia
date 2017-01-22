package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDivmod(t *testing.T) {
	as := []int{0, 1, 2, 3, 4, 5}
	bs := []int{1, 2, 3, 3, 2, 3}
	qs := []int{0, 0, 0, 1, 2, 1}
	rs := []int{0, 1, 2, 0, 0, 2}

	for i, a := range as {
		b := bs[i]
		eq := qs[i]
		er := rs[i]

		q, r := divmod(a, b)
		if q != eq {
			t.Errorf("Expected %d / %d to equal %d. Got %d instead", a, b, eq, q)
		}
		if r != er {
			t.Errorf("Expected %d %% %d to equal %d. Got %d instead", a, b, er, r)
		}
	}

	assert := assert.New(t)
	fail := func() {
		divmod(1, 0)
	}
	assert.Panics(fail)
}
