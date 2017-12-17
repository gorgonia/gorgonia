# Contributing #

We want to make contributing as easy as possible. There are a number of [issues](https://github.com/gorgonia/gorgonia/issues) that can be solved. Most of the issues are labelled.

## Labels ##

Here's the current explanation of the labels:

<table>
<thead><th>Label</th><th>Expertise Required</th><th>Notes</th></thead>
<tbody>
<tr><td>easy</td><td>Familiar with Go</td><td>Usually there would be a link to an existing implementation that is similar</td></tr>
<tr><td>requires Linear Algebra knowledge</td><td>Linear algebra knowledge required on top of being familiar with Go</td><td>Linear Algebra knowledge will go a long way in helping identify subtler bugs</td></tr>
<tr><td>no label</td><td>Knowledge about Gorgonia</td><td></td></tr>
<tr><td>help wanted</td><td>Various expertise</td><td>Typically it means the task requires extranormal speciality that the author doesn't possess</td></tr>
<tr><td>complicated</td><td>-</td><td>The code change will affect, and involve many files</td></tr>
</tbody>
</table>

## Steps ##
1. Fork this project on Github
2. Clone to your local drive
3. Check if there are any pending issues in the issues tracker
4. Pick an unassigned issue that you can accomplish. Comment on the issue to pick it up.
5. Work on it, using topic branches is highly recommended.

## Testing ##

Testing is important


## How To Get Your Pull Request Accepted ##

1. Test, test, test. Make sure your new code doesn't break the existing tests
2. If you add new code, you must add tests.
3. `gofmt` your code
5. Atomic pull requests - solve one issue per pull request. Some issues may break down to atomic tasks. In those cases, it's OK to solve them partially.


## Git Workflow ##

The master branch is considered to be the "canonical" branch. There is no develop branch. The author prefers use of topic branches. The workflow can best be described by the [Github Flow](https://guides.github.com/introduction/flow/). Please try to keep to this flow.

# Development #

## How the `Tensor` Libs Are Developed ##


## Debugging ##

Whilst the author encourages the use of [Delve](https://github.com/derekparker/delve), it may often be easier to log the trace using the debug loggers. Gorgonia comes with a debug build tag precisely to help with that. To build debug builds, simply do this:

```go
go build -tags='debug' . 
```

The debug tag enables various tracing options, available in `debug.go`. There are several debug constants that are used:

* `compileDev`       
* `shapeInferenceDev`
* `typeSystemDev`    
* `symdiffDev`       
* `autodiffDev`      
* `machineDev`       
* `stabilizationDev` 
* `solverDev`
* `cudaDev`        

These are the bools that you need to set in order to get a trace. If for example, you think there is something wrong with the type system, simply set `typeSystemDev` to `true` and then insert `typeSysLogf` into wherever you want to trace. 

