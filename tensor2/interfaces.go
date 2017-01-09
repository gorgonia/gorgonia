package tensor

// Eq is any type where you can perform an equality test
type Eq interface {
	Eq(interface{}) bool
}

// Boolable is any type has a zero and one value, and is able to set itself to either
type Boolable interface {
	Zeroer
	Oner
}

// A Zeroer is any type that can set itself to the zeroth value. It's used to implement the arrays
type Zeroer interface {
	Zero()
}

// A Oner is any type that can set itself to the equivalent of one. It's used to implement the arrays
type Oner interface {
	One()
}

// A MemSetter is any type that can set itself to a value.
type MemSetter interface {
	Memset(interface{}) error
}

// ArrayMaker is for custom Dtypes
type ArrayMaker interface {
	MakeArray(size int) Array
}

// FromInterfaceSlicer is for custom Dtypes
type FromInterfaceSlicer interface {
	FromInterfaceSlice(s []interface{}) Array
}

// CopierFrom copies from source to the receiver. It returns an int indicating how many bytes have been copied
type CopierFrom interface {
	CopyFrom(src interface{}) (int, error)
}

// CopierTo copies from the receiver to the dest. It returns an int indicating how many bytes have been copied
type CopierTo interface {
	CopyTo(dest interface{}) (int, error)
}
