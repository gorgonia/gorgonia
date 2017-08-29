// +build !go1.9

package tensor

func clz(a uint) (retVal int) {
	for a != 0 {
		retVal++
		a>>=1
	}
	return
}

func clz64(a uint64) (retVal int) {
	for a != 0 {
		retVal++
		a>>=1
	}
	return
}
func clz32(a uint32) (retVal int) {
	for a != 0 {
		retVal++
		a >>= 1
	}
	return
}

func clz16(a uint16) (retVal int) {
	for a != 0 {
		retVal++
		a>>=1
	}
	return
}


func clz8(a uint8) (retVal int) {
	for a != 0 {
		retVal++
		a>>=1
	}
	return
}
