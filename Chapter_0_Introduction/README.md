# Hello CUDA — Minimal Kernel Demo

This tiny program verifies your CUDA setup by compiling and launching the smallest possible GPU kernel that prints a message. It is a safe first step to confirm that your compiler, driver, and device are working together.

## What this program does

* Defines a GPU kernel `helloGPU` that prints a line.
* Launches the kernel with one block of one thread: `<<<1,1>>>`.
* Synchronizes the device so the message flushes before the program exits.

Expected console output:

```
Hello from GPU!
```

## Prerequisites

* Windows 10/11 (64-bit)
* NVIDIA CUDA-capable GPU with a recent NVIDIA driver
* CUDA Toolkit installed (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)
* Microsoft Visual Studio 2019 or 2022 with **Desktop development with C++**
* A terminal you can build from:

  * **x64 Native Tools Command Prompt for VS 2019/2022** (recommended), or
  * **Developer PowerShell for VS**, or
  * Command Prompt with VS and CUDA in your `PATH`

## Files

* `hello.cu` — the CUDA source file

```cpp
#include <stdio.h>

__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    helloGPU<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

## Build and run (quick start)

1. Open **x64 Native Tools Command Prompt** for your Visual Studio version.
2. Change directory to where `hello.cu` lives.
3. Compile with `nvcc` and run:

```cmd
nvcc hello.cu -o hello.exe
hello.exe
```

You should see:

```
Hello from GPU!
```

## If `nvcc` is not found

* Open a new terminal after installing CUDA so updated environment variables load.
* Verify the CUDA `bin` folder is in `PATH`. For example:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
```

* Check `nvcc --version` prints a version string.

## Building in Visual Studio (optional)

1. **File → New → Project → Console App** (C++).
2. Add `hello.cu` to the project.
3. Right-click the project → **Build Dependencies → Build Customizations…**
   Check **CUDA x.y** to enable CUDA for the project.
4. Right-click the project → **Properties**:

   * **Configuration Properties → CUDA C/C++** should appear.
   * Ensure **C/C++ → Language → C++ Language Standard** is a supported standard (e.g., `/std:c++17`).
5. Build and run (Ctrl+F5). Output should match the quick start.

## Common issues and quick fixes

* **Program compiles but prints nothing**
  Add `cudaDeviceSynchronize()` after the kernel launch so the CPU waits for GPU output.

* **Linker errors referencing CUDA runtime**
  Ensure CUDA build customizations are enabled, or compile with `nvcc` instead of `cl.exe`.

* **`device not found` or similar runtime errors**
  Update your NVIDIA driver to a version compatible with your CUDA Toolkit.

* **Multiple Visual Studio versions installed**
  Use the **x64 Native Tools Command Prompt** matching the VS version you will use to build.

## How this verifies your setup

* Confirms the CUDA compiler (`nvcc`) is reachable.
* Confirms the runtime can launch a kernel on your GPU.
* Confirms stdout from device code reaches the host console.
* Confirms driver and toolkit versions are compatible enough to run a kernel.

## Next steps

* Increase the launch configuration to print from many threads and observe interleaving:

  ```cpp
  helloGPU<<<1, 8>>>();
  ```
* Try passing data from host to device and back (vector scale).
* Explore CUDA samples in:

  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\samples
  ```
* Read about grids, blocks, and threads to map work to data effectively.

