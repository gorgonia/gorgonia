package main

import (
	"fmt"
	"image"
	"sort"

	"gorgonia.org/tensor"
)

// DetectionRectangle Representation of detection
type DetectionRectangle struct {
	conf  float32
	rect  image.Rectangle
	class string
	score float32
}

func (dr *DetectionRectangle) String() string {
	return fmt.Sprintf("Detection:\n\tClass = %s\n\tScore = %f\n\tConfidence = %f\n\tCoordinates: [RightTopX = %d, RightTopY = %d, LeftBottomX = %d, LeftBottomY = %d]",
		dr.class, dr.score, dr.conf, dr.rect.Min.X, dr.rect.Min.Y, dr.rect.Max.X, dr.rect.Max.Y,
	)
}

// GetClass Get class of object
func (dr *DetectionRectangle) GetClass() string {
	return dr.class
}

// Detections Detection rectangles
type Detections []*DetectionRectangle

func (detections Detections) Len() int { return len(detections) }
func (detections Detections) Swap(i, j int) {
	detections[i], detections[j] = detections[j], detections[i]
}
func (detections Detections) Less(i, j int) bool { return detections[i].conf < detections[j].conf }

// DetectionsOrder Ordering for X-axis
type DetectionsOrder []*DetectionRectangle

func (detections DetectionsOrder) Len() int { return len(detections) }
func (detections DetectionsOrder) Swap(i, j int) {
	detections[i], detections[j] = detections[j], detections[i]
}
func (detections DetectionsOrder) Less(i, j int) bool {
	return detections[i].rect.Min.X < detections[j].rect.Min.X
}

// ProcessOutput Detection layer
func (net *YOLOv3) ProcessOutput(classes []string, scoreTreshold, iouTreshold float32) (Detections, error) {
	if len(classes) != net.classesNum {
		return nil, fmt.Errorf("length of provided slice of classes is not equal to YOLO network 'classesNum' field")
	}

	preparedDetections := make(Detections, 0)
	out := net.GetOutput()
	for i := range out {
		nodeValue := out[i].Value()
		var tensorValue tensor.Tensor
		switch nodeValue.(type) {
		case tensor.Tensor:
			tensorValue = nodeValue.(tensor.Tensor)
			break
		default:
			fmt.Printf("Warning: YOLO output node #%d should be type of tensor.Tensor", i)
			break
		}

		dataValue := tensorValue.Data()
		dataF32 := make([]float32, 0)
		switch dataValue.(type) {
		case []float32:
			dataF32 = dataValue.([]float32)
			break
		case []float64:
			dataF64 := dataValue.([]float64)
			dataF32 = make([]float32, len(dataF64))
			for d := range dataF64 {
				dataF32[d] = float32(dataF64[d])
			}
			break
		default:
			fmt.Printf("Warning: YOLO output tensor #%d should be type of []float32 or []float64", i)
			break
		}

		detections := prepareDetections(dataF32, scoreTreshold, net.netSize, classes)
		preparedDetections = append(preparedDetections, detections...)
	}

	finalDetections := nonMaxSupr(preparedDetections, iouTreshold)
	sort.Sort(DetectionsOrder(finalDetections))
	return finalDetections, nil
}

func prepareDetections(data []float32, scoreTreshold float32, netSize int, classes []string) Detections {
	detections := make(Detections, 0)
	for i := 0; i < len(data); i += (len(classes) + 5) {
		class := 0
		var maxProbability float32
		for j := 5; j < 5+len(classes); j++ {
			if data[i+j] > maxProbability {
				maxProbability = data[i+j]
				class = (j - 5) % len(classes)
			}
		}
		if maxProbability*data[i+4] > scoreTreshold {
			box := &DetectionRectangle{
				conf:  data[i+4],
				rect:  Rectify(int(data[i]), int(data[i+1]), int(data[i+2]), int(data[i+3]), netSize, netSize),
				class: classes[class],
				score: maxProbability,
			}
			detections = append(detections, box)
		}
	}
	return detections
}

// nonMaxSupr Sorts boxes by confidence
func nonMaxSupr(detections Detections, iouTreshold float32) Detections {
	sort.Sort(detections)
	nms := make(Detections, 0)
	if len(detections) == 0 {
		return nms
	}
	nms = append(nms, detections[0])
	for i := 1; i < len(detections); i++ {
		tocheck, del := len(nms), false
		for j := 0; j < tocheck; j++ {
			currIOU := IOUFloat32(detections[i].rect, nms[j].rect)
			if currIOU > iouTreshold && detections[i].class == nms[j].class {
				del = true
				break
			}
		}
		if !del {
			nms = append(nms, detections[i])
		}
	}
	return nms
}
