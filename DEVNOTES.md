# Cgo #

Cgo is used pretty heavily in Gorgonia. Here are some Cgo guidelines that I've come up with and used over the years:

1. Use astyle to fmt your C code. This is the command to use: `astyle --style=google -lineend=linux --indent=tab --indent-switches --align-pointer=type --align-reference=name --delete-empty-lines`. Yes, the choices are  a bit weird, but it makes C more like Go code, which is readable af.
2. When passing Go slices to a C function, pass a splatted [fat pointer](http://www.drdobbs.com/architecture-and-design/cs-biggest-mistake/228701625). What I mean by this is to do something like this (cap is optional but recommended):

	```c
	void foo(double* sliceF64, int len, int cap) {

	}
	```
3. Brackets are your friends. It's tempting to write this:
	```c
	if (foo)
		bar()
	```.
	Don't. Write shis instead:
	```c
	if (foo) {
		bar()
	}
	```

# Go Assembly #

When writing Go Assembly, use [asmfmt](https://github.com/klauspost/asmfmt)


# General Patterns #

This section describes the general patterns preferred in this library.

## API ##

The API of this library exposes functions to users of this library. API functions and methods should return `error` whenever an error may occur so the user of the library will be able to handle it on their own.

## Kernels ##

Functions that return exactly one thing (i.e. no errors) are called *kernels*. It is preferable to write your API functions so they conform to this pattern:


```
func Foo(x, y, z T) (retVal T, err error) {
	// checks for errors
	if err := check(x, y, z); err != nil {
		return nil, err
	}

	// from this point on, there should be no errors returned.

	// do thing
	retVal = doThing(x, y, z)
	return retVal, nil
}
```

Here, we see `Foo` being actually composed of two other functions: `check` and `doThing`. `doThing` is considered the kernel.

The reason for preferring this is that we can reuse the kernel functions when it comes to optimizing code.

**Moral**: where possible, abstract all checking and error returning things into another function. This allows the kernel to be used in tighter loops when it comes to optimization.
