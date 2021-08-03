package cudalib

import (
	"sync"
)

type Lib struct {
	ModuleName string
	Data       string
	Funcs      []string
}

var lock sync.Mutex
var stdlib []Lib

func AddToStdLib(name, data string, funcs []string) {
	lock.Lock()
	stdlib = append(stdlib, Lib{
		ModuleName: name,
		Data:       data,
		Funcs:      funcs,
	})
	lock.Unlock()
}

func StandardLib() []Lib {
	lock.Lock()
	retVal := make([]Lib, len(stdlib))
	copy(retVal, stdlib)
	lock.Unlock()
	return retVal
}
