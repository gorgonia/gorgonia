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

#Go Assembly#

When writing Go Assembly, use [asmfmt](https://github.com/klauspost/asmfmt)