/*
You must have ffmpeg and OpenCV installed in order to run this code. It will connect to the Tello
and then open a window using OpenCV showing the streaming video.

How to run

	go run ./main.go ./res10_300x300_ssd_iter_140000.caffemodel ./deploy.prototxt opencv cpu
*/

package main

import (
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"
	"time"

	"gobot.io/x/gobot"
	"gobot.io/x/gobot/platforms/dji/tello"
	"gocv.io/x/gocv"
)

type pair struct {
	x float64
	y float64
}

const (
	frameX    = 400
	frameY    = 300
	frameSize = frameX * frameY * 3
	offset    = 32767.0
)

var (
	// ffmpeg command to decode video stream from drone
	ffmpeg = exec.Command("ffmpeg", "-hwaccel", "auto", "-hwaccel_device", "opencl", "-i", "pipe:0",
		"-pix_fmt", "bgr24", "-s", strconv.Itoa(frameX)+"x"+strconv.Itoa(frameY), "-f", "rawvideo", "pipe:1")
	ffmpegIn, _  = ffmpeg.StdinPipe()
	ffmpegOut, _ = ffmpeg.StdoutPipe()

	// gocv
	window = gocv.NewWindow("Tello")
	net    *gocv.Net
	green  = color.RGBA{0, 255, 0, 0}

	// tracking
	tracking                 = false
	detected                 = false
	detectSize               = false
	distTolerance            = 0.05 * dist(0, 0, frameX, frameY)
	refDistance              float64
	left, top, right, bottom float64

	// drone
	drone      = tello.NewDriver("8890")
	flightData *tello.FlightData
)

func init() {
	// process drone events in separate goroutine for concurrency
	go func() {
		if err := ffmpeg.Start(); err != nil {
			fmt.Println(err)
			return
		}

		drone.On(tello.FlightDataEvent, func(data interface{}) {
			// TODO: protect flight data from race condition
			flightData = data.(*tello.FlightData)
		})

		drone.On(tello.ConnectedEvent, func(data interface{}) {
			fmt.Println("Connected")
			drone.TakeOff()

			drone.StartVideo()
			drone.SetVideoEncoderRate(tello.VideoBitRateAuto)
			drone.SetExposure(0)
			gobot.Every(100*time.Millisecond, func() {
				drone.StartVideo()
			})
		})

		drone.On(tello.VideoFrameEvent, func(data interface{}) {
			pkt := data.([]byte)
			if _, err := ffmpegIn.Write(pkt); err != nil {
				fmt.Println(err)
			}
		})

		robot := gobot.NewRobot("tello",
			[]gobot.Connection{},
			[]gobot.Device{drone},
		)

		robot.Start()
	}()
}

func main() {
	if len(os.Args) < 5 {
		fmt.Println("How to run:\ngo run facetracker.go [model] [config] ([backend] [device])")
		return
	}

	model := os.Args[1]
	config := os.Args[2]
	backend := gocv.NetBackendDefault
	if len(os.Args) > 3 {
		backend = gocv.ParseNetBackend(os.Args[3])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 4 {
		target = gocv.ParseNetTarget(os.Args[4])
	}

	n := gocv.ReadNet(model, config)
	if n.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, config)
		return
	}
	net = &n
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	for {
		// get next frame from stream
		buf := make([]byte, frameSize)
		if _, err := io.ReadFull(ffmpegOut, buf); err != nil {
			fmt.Println(err)
			continue
		}
		img, _ := gocv.NewMatFromBytes(frameY, frameX, gocv.MatTypeCV8UC3, buf)
		if img.Empty() {
			continue
		}

		trackFace(&img)

		window.IMShow(img)
		if window.WaitKey(10) >= 0 {
			break
		}
	}
}

func trackFace(frame *gocv.Mat) {
	W := float64(frame.Cols())
	H := float64(frame.Rows())

	blob := gocv.BlobFromImage(*frame, 1.0, image.Pt(300, 300), gocv.NewScalar(104, 177, 123, 0), false, false)
	defer blob.Close()

	net.SetInput(blob, "data")

	detBlob := net.Forward("detection_out")
	defer detBlob.Close()

	detections := gocv.GetBlobChannel(detBlob, 0, 0)
	defer detections.Close()

	for r := 0; r < detections.Rows(); r++ {
		confidence := detections.GetFloatAt(r, 2)
		if confidence < 0.5 {
			continue
		}

		left = float64(detections.GetFloatAt(r, 3)) * W
		top = float64(detections.GetFloatAt(r, 4)) * H
		right = float64(detections.GetFloatAt(r, 5)) * W
		bottom = float64(detections.GetFloatAt(r, 6)) * H

		left = math.Min(math.Max(0.0, left), W-1.0)
		right = math.Min(math.Max(0.0, right), W-1.0)
		bottom = math.Min(math.Max(0.0, bottom), H-1.0)
		top = math.Min(math.Max(0.0, top), H-1.0)

		detected = true
		rect := image.Rect(int(left), int(top), int(right), int(bottom))
		gocv.Rectangle(frame, rect, green, 3)
	}

	if !tracking || !detected {
		return
	}

	if detectSize {
		detectSize = false
		refDistance = dist(left, top, right, bottom)
	}

	distance := dist(left, top, right, bottom)

	// x axis
	switch {
	case right < W/2:
		drone.CounterClockwise(50)
	case left > W/2:
		drone.Clockwise(50)
	default:
		drone.Clockwise(0)
	}

	// y axis
	switch {
	case top < H/10:
		drone.Up(25)
	case bottom > H-H/10:
		drone.Down(25)
	default:
		drone.Up(0)
	}

	// z axis
	switch {
	case distance < refDistance-distTolerance:
		drone.Forward(20)
	case distance > refDistance+distTolerance:
		drone.Backward(20)
	default:
		drone.Forward(0)
	}
}

func dist(x1, y1, x2, y2 float64) float64 {
	return math.Sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
}
