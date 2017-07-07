package tensor

type binOpTest struct {
	a, b interface{}
	op func(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error)
}