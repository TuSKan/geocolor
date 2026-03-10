package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/TuSKan/geocolor"
)

func currentDir() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Dir(file)
}

// Structures to hold our new polygonal geometry
type Vertex struct{ X, Y, Z, R, G, B float64 }
type Face struct{ V1, V2, V3 int }

func main() {
	// 1. Define High-Resolution Configuration
	grid := &geocolor.GridConfig{
		ResL: 100,
		ResA: 301, // Chroma (Radius)
		ResB: 301, // Hue (Angle)

		// L: 0 to 100 (Black to White)
		// C: 0.0 to 150.0 (Neutral center to max saturation)
		// H: -math.Pi to math.Pi (Full 360 degree circle)
		MinBounds: [3]float64{0.0, 0.0, -math.Pi},
		MaxBounds: [3]float64{100.0, 150.0, math.Pi},
	}

	cParam := float64(0.01) // Relaxed dampening curve

	// 2. Initialize and Compute
	log.Println("Compiling and executing XLA tensor graph...")
	cs := geocolor.NewColorSpace(grid, cParam)

	if err := cs.Compute(); err != nil {
		log.Fatal(err)
	}

	// Assuming you have fetched both 4D slices from your GoMLX results
	// labVals shape: [ResL][ResA][ResB][3]
	// rgbVals shape: [ResL][ResA][ResB][3]
	lchVals := cs.GetLCHColors()
	rgbVals := cs.GetRGBColors()

	// Assuming ResL=100, ResA=301, ResB=301
	// L=50 is index 50. a=0 and b=0 are at index 150.
	if len(rgbVals) > 50 && len(rgbVals[0]) > 150 {
		centerColor := rgbVals[50][150][150]
		log.Printf("DEBUG - Neutral Axis (L=50, a=0, b=0) -> R:%f, G:%f, B:%f",
			centerColor[0], centerColor[1], centerColor[2])
	}
	if len(lchVals) > 50 && len(lchVals[0]) > 150 {
		inputPos := lchVals[50][150][150]
		log.Printf("DEBUG - Input Voxel (Index 50, 150, 150) -> L:%f, C:%f, H:%f",
			inputPos[0], inputPos[1], inputPos[2])
	}

	var vertices []Vertex
	var faces []Face

	log.Println("Extracting Triangulated Starburst...")

	numPetals := 15
	hueStep := grid.ResB / numPetals
	lStep := 1
	cStep := 1

	for petal := 0; petal < numPetals; petal++ {
		targetHueIdx := petal * hueStep

		vMap := make([][]int, grid.ResL)
		for i := range vMap {
			vMap[i] = make([]int, grid.ResA)
			for j := range vMap[i] {
				vMap[i][j] = -1 // -1 means out of gamut
			}
		}

		// 1. Extract Vertices
		for lIdx := 0; lIdx < grid.ResL; lIdx += lStep {
			for cIdx := 0; cIdx < grid.ResA; cIdx += cStep {
				voxelColor := rgbVals[lIdx][cIdx][targetHueIdx]
				r, g, bl := float64(voxelColor[0]), float64(voxelColor[1]), float64(voxelColor[2])

				if r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && bl >= 0.0 && bl <= 1.0 {
					voxelPos := lchVals[lIdx][cIdx][targetHueIdx]
					L_val := float64(voxelPos[0])
					C_val := float64(voxelPos[1])
					H_val := float64(voxelPos[2])

					// Wrap the cylinder into 3D Cartesian space
					X_pos := C_val * math.Cos(H_val)
					Z_pos := C_val * math.Sin(H_val)

					vMap[lIdx][cIdx] = len(vertices)

					vertices = append(vertices, Vertex{
						X: X_pos,
						Y: L_val,
						Z: Z_pos,
						R: r * 255.0, G: g * 255.0, B: bl * 255.0,
					})
				}
			}
		}

		// 2. Stitch Faces
		for lIdx := 0; lIdx < grid.ResL-lStep; lIdx += lStep {
			for cIdx := 0; cIdx < grid.ResA-cStep; cIdx += cStep {
				v00 := vMap[lIdx][cIdx]
				v10 := vMap[lIdx+lStep][cIdx]
				v01 := vMap[lIdx][cIdx+cStep]
				v11 := vMap[lIdx+lStep][cIdx+cStep]

				if v00 != -1 && v10 != -1 && v01 != -1 {
					faces = append(faces, Face{V1: v00, V2: v10, V3: v01})
				}
				if v10 != -1 && v11 != -1 && v01 != -1 {
					faces = append(faces, Face{V1: v10, V2: v11, V3: v01})
				}
			}
		}
	}

	log.Println("Writing Triangulated Mesh to PLY file...")
	if err := ExportMeshToPLY(filepath.Join(currentDir(), "geocolor.ply"), vertices, faces); err != nil {
		log.Fatalf("Failed to export PLY: %v", err)
	}
	log.Println("Done! Drop geocolor.ply into MeshLab.")
}

// ExportPointCloudToPLY writes an interleaved [X,Y,Z, R,G,B] float64 slice to a PLY file.
// Expects XYZ as spatial coordinates and RGB as 0.0 - 1.0 normalized floats.
func ExportPointCloudToPLY(filename string, vertexData []float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Use buffered writer for massive performance gains on large point clouds
	writer := bufio.NewWriter(file)

	// A single vertex consists of 6 floats (X, Y, Z, R, G, B)
	numVertices := len(vertexData) / 6

	// 1. Write PLY Header
	writer.WriteString("ply\n")
	writer.WriteString("format ascii 1.0\n")
	writer.WriteString(fmt.Sprintf("element vertex %d\n", numVertices))
	writer.WriteString("property float x\n")
	writer.WriteString("property float y\n")
	writer.WriteString("property float z\n")
	writer.WriteString("property uchar red\n") // Colors in PLY are typically 0-255 ints
	writer.WriteString("property uchar green\n")
	writer.WriteString("property uchar blue\n")
	writer.WriteString("end_header\n")

	// 2. Write Vertices
	for i := 0; i < len(vertexData); i += 6 {
		x := vertexData[i]
		y := vertexData[i+1]
		z := vertexData[i+2]

		// Convert normalized 0.0-1.0 floats to 8-bit integers (0-255)
		r := uint8(vertexData[i+3])
		g := uint8(vertexData[i+4])
		b := uint8(vertexData[i+5])

		// Format: X Y Z R G B
		fmt.Fprintf(writer, "%f %f %f %d %d %d\n", x, y, z, r, g, b)
	}

	// 3. Flush the buffer to disk
	return writer.Flush()
}

// ExportMeshToPLY writes both vertices and triangulated faces to a PLY file.
func ExportMeshToPLY(filename string, vertices []Vertex, faces []Face) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	fmt.Fprintln(writer, "ply")
	fmt.Fprintln(writer, "format ascii 1.0")
	fmt.Fprintf(writer, "element vertex %d\n", len(vertices))
	fmt.Fprintln(writer, "property float x")
	fmt.Fprintln(writer, "property float y")
	fmt.Fprintln(writer, "property float z")
	fmt.Fprintln(writer, "property uchar red")
	fmt.Fprintln(writer, "property uchar green")
	fmt.Fprintln(writer, "property uchar blue")
	fmt.Fprintf(writer, "element face %d\n", len(faces))
	fmt.Fprintln(writer, "property list uchar int vertex_index")
	fmt.Fprintln(writer, "end_header")

	for _, v := range vertices {
		fmt.Fprintf(writer, "%f %f %f %d %d %d\n",
			v.X, v.Y, v.Z, uint8(v.R), uint8(v.G), uint8(v.B),
		)
	}

	for _, f := range faces {
		fmt.Fprintf(writer, "3 %d %d %d\n", f.V1, f.V2, f.V3)
	}

	return writer.Flush()
}
