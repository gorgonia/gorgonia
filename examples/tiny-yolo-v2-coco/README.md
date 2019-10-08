# Tiny YOLO v2 #

This is an example of Tiny YOLO v2 neural network. You can read about this network [here](https://pjreddie.com/darknet/yolov2/).

Folder `model` contains yolov2-tiny.cfg on which file `feedforward.go` based.

Folder `data` contains image file `dog_416x416.jpg` - this is scaled to 416x416 image for make it better understanding of how net works.

For now this example works not quite fast (it is only "tiny" version). Probably it is better to implement this network using CUDA.

How to run:
```go
go run .
```
