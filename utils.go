package gorgonia

import (
	"fmt"
	"sync"
)

var rndCounter int
var rndLock sync.Mutex

func randomName(a Tensor) string {
	rndLock.Lock()
	defer rndLock.Unlock()
	rndCounter++
	return fmt.Sprintf("Random_%d", rndCounter)
}
