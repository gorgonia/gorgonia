package tensor

// Eq is any type where you can perform an equality test
type Eq interface {
	Eq(interface{}) bool
}

// Cloner is any type that can clone itself
type Cloner interface {
	Clone() interface{}
}

// Dataer is any type that returns the data in its original form (typically a Go slice of something)
type Dataer interface {
	Data() interface{}
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
