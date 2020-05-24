# dual #

Package `dual` provides dual numbers for Gorgonia. Dual numbers are useful in automatic differentiation. However, some exceptions apply. For more information on dual numbers and their properties, see https://en.wikipedia.org/wiki/Dual_number.

If you are looking for a pure `float64` dual number package, take a look at [gonum.org/v1/gonum/num/dual](https://gonum.org/v1/gonum/num/dual).

# Implementation #

This package provides:

* The data structure of a dual number.
* Functions to deal with the data structures.

A `*dual.Dual` is a tuple of two values, defined as follows:

```
type Dual struct {
	values.Value
	d values.Value
}
```

The preferred way to use a `dual.Dual` is as a pointer to an object - i.e. `*dual.Dual`. This is because of the interfaces a `*dual.Dual` implements. Specifically, a `*dual.Dual` implements `value.Value`.

When creating a new dual number (using `New`, `NewVar`, `Bind`, `BindVar`, `Bind0` etc), a pointer value is returned. This causes the value to be allocated on the heap. Thus, when reflecting or type asserting from an interface, no additional allocations are needed.

Because Gorgonia itself does a lot of type switches on the types of values, the preferred way is to use pointers to the tuple itself.


# On the API #

The API of this package may seem a little abnormal to Gophers. Here the reasons are listed.

To construct a new `*Dual`, there are several options:

* `New`
* `NewVar`

These functions take a `values.Value` and returns a new `*Dual`. With some exceptions.

Because a `*Dual` is itself a `values.Value`, if a `*Dual` is passed in, these functions will return the input argumet.

```go
// create a new value
var a Value = values.NewF64(3.14)

// create a new *Dual from a.
d := New(a)

// create a new *Dual from d.
d2 := New(d)

d2 == d // true
```

## Where Are The Methods/Arithmetical Functions? ##

If you are used to dual numbers, for example you might expect that these numbers support basic aritmetics.

<details>
<summary>Example (click to expand)</summary>

In this example, we will use the syntax as laid out in Wikipedia. Multiplication is given as such:

![(a+b\varepsilon)(c+d\varepsilon) = ac + (ad+bc)\varepsilon ](https://render.githubusercontent.com/render/math?math=(a%2Bb%5Cvarepsilon)(c%2Bd%5Cvarepsilon)%20%3D%20ac%20%2B%20(ad%2Bbc)%5Cvarepsilon%20)

Thus the equivalent using this package, we will get:

```
var a, b values.Value
dn1 := &Dual {
	Value: a,
	d: b,
}


var c, d values.Value
dn2 := &Dual{
	Value: c,
	d: d,
}

dn3 := Mul(dn1, d2)
```

The problem of course, is that `Mul` does not exist in our API. In fact, one cannot construct `*Dual` from a literal like in the example either!

</details>

The reason why there are no aritmetical functions listed in the API is because of the way we intend to use the `*Dual`.

The `*Dual` is used both as a dual number, and as a storage for gradients.

## How To Use `*Dual`? ##

Or, what is this `Bind`, `BindVar` and `Bind0` business all about?

As it stands, a `*Dual` is a dumb data structure with no smarts around it. Instead, we must imbue it with some smarts.

The following example shows the same multiplication example from the section above.

```
func extract2Singles(vals ...values.Value)(retVal [2]float64){
	retVal[0] = *(vals[0].Value.(*values.F64))
	retVal[1] = *(vals[1].Value.(*values.F64))
	return
}

func singlesMul(vals ...values.Value) values.Value{
	vs := extract2singles(vals)
	return values.NewF64(vs[0]*vs[1])
}

func Mul(vals ...values.Value)(values.Value, error){
	ds, ok := checkDuals(vals...)
	if !ok {
		return singlesMul(vals...)
	}

	a := *(ds[0].Value.(*values.F64))
	bɛ := *(ds[0].Deriv().(*values.F64))
	c := *(ds[1].Value.(*values.F64))
	dɛ := *(ds[1].Deriv().(*values.F64))

	e := a * b
	f := a * dɛ + c * bɛ

	retVal := new(dual.Dual)
	if err := retVal.SetValue(e); err != nil{
		return nil, err
	}
	if err := retVal.SetDeriv(f); err != nil{
		return nil, err
	}
	return retVal, nil
}
```

This seems rather involved!
