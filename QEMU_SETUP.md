# ARM QEMU Setup for Post 2 Labs

**Purpose:** Install QEMU and boot an AArch64 VM to run the Post 2 labs with real ACfL/ArmPL output instead of mock data.

**Prerequisite:** Python 3.11+, `pip install -r requirements.txt` already done on your host machine.

---

## Which labs benefit from QEMU?

| Lab | Benefit | What changes on real ARM |
|-----|---------|--------------------------|
| Lab 1 — ACfL Flags Explorer | **HIGH** | Real `armclang` vectorization report replaces `acfl_sve2_mock_output.txt` |
| Lab 2 — ArmPL BLAS Explorer | **HIGH** | Real DGEMM/FFT benchmarks via ctypes replace synthetic JSON |
| Lab 3 — Slurm Job Architect | **LOW** | Needs a real Slurm cluster, not just a single QEMU VM |
| Lab 4 — Linaro Forge MAP | **LOW** | Needs Linaro Forge license + cluster for meaningful profiles |

All 4 labs run fully on macOS with mock data. QEMU is optional — use it when you want real compiler output or real library benchmarks.

---

## User-mode vs system emulation

QEMU offers two modes for ARM:

- **`qemu-aarch64` (user-mode):** Translates individual AArch64 binaries on your host kernel. Fast, no VM image needed, but cannot run a full OS — you cannot install ACfL, run `module load`, or host JupyterLab. Not suitable for these labs.
- **`qemu-system-aarch64` (system):** Boots a complete AArch64 Linux VM with its own kernel, package manager, and userspace. Slower, requires a disk image, but provides the full environment these labs need.

This guide uses **system emulation** exclusively.

**Performance note:** On Apple Silicon Macs, QEMU uses Hypervisor.framework (`-accel hvf`) for AArch64-on-AArch64 — near-native speed. On Intel Macs and x86_64 Linux, expect 5-20x slowdown from full software translation.

---

## 1. Install QEMU

### macOS (Homebrew)

```bash
brew install qemu
qemu-system-aarch64 --version
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y qemu-system-arm qemu-efi-aarch64 cloud-image-utils
```

### Fedora / RHEL

```bash
sudo dnf install -y qemu-system-aarch64 edk2-aarch64
```

Verify: `qemu-system-aarch64 --version` should report 8.x or 9.x.

---

## 2. Get an AArch64 VM image

### Download Ubuntu 24.04 LTS (arm64)

```bash
# Create a working directory
mkdir -p ~/arm-qemu && cd ~/arm-qemu

# Download the cloud image
curl -LO https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img

# Expand the disk to 20 GB (default is ~3 GB, too small for ACfL)
qemu-img resize noble-server-cloudimg-arm64.img 20G
```

### Create a cloud-init seed ISO (sets your login credentials)

```bash
# Create cloud-init config files
cat > user-data <<'EOF'
#cloud-config
password: armlabs
chpasswd:
  expire: false
ssh_pwauth: true
EOF

cat > meta-data <<'EOF'
instance-id: arm-labs-vm
local-hostname: arm-labs
EOF
```

**macOS** (create seed ISO):
```bash
mkisofs -output seed.iso -volid cidata -joliet -rock user-data meta-data
```

If `mkisofs` is not available: `brew install cdrtools`

**Linux** (create seed ISO):
```bash
cloud-localds seed.iso user-data meta-data
```

If `cloud-localds` is not available: `sudo apt install cloud-image-utils`

---

## 3. Locate the UEFI firmware

QEMU needs AArch64 UEFI firmware to boot the VM. The path depends on your install method.

**macOS (Homebrew):**
```bash
# Typically at:
UEFI_FW="$(brew --prefix)/share/qemu/edk2-aarch64-code.fd"
ls -la "$UEFI_FW"
```

**Ubuntu / Debian:**
```bash
UEFI_FW="/usr/share/AAVMF/AAVMF_CODE.fd"
# or: /usr/share/qemu-efi-aarch64/QEMU_EFI.fd
ls -la "$UEFI_FW"
```

**Fedora / RHEL:**
```bash
UEFI_FW="/usr/share/edk2/aarch64/QEMU_EFI.fd"
ls -la "$UEFI_FW"
```

---

## 4. Boot the VM

### Apple Silicon Mac (hardware-accelerated)

```bash
qemu-system-aarch64 \
  -M virt \
  -accel hvf \
  -cpu host \
  -m 4096 \
  -smp 4 \
  -bios "$UEFI_FW" \
  -drive file=noble-server-cloudimg-arm64.img,format=qcow2,if=virtio \
  -drive file=seed.iso,format=raw,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8888-:8888 \
  -device virtio-net-pci,netdev=net0 \
  -nographic
```

### Intel Mac / x86_64 Linux (software emulation)

```bash
qemu-system-aarch64 \
  -M virt \
  -cpu neoverse-v1 \
  -m 4096 \
  -smp 4 \
  -bios "$UEFI_FW" \
  -drive file=noble-server-cloudimg-arm64.img,format=qcow2,if=virtio \
  -drive file=seed.iso,format=raw,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8888-:8888 \
  -device virtio-net-pci,netdev=net0 \
  -nographic
```

**Key flags:**
- `-cpu neoverse-v1`: Highest Neoverse model QEMU supports (as of QEMU 9.x). V3 is not available.
- `-cpu host` (Apple Silicon only): Exposes the host CPU directly via HVF — fastest option.
- Port forwarding: SSH on `localhost:2222`, JupyterLab on `localhost:8888`.

**First boot** takes 1-2 minutes. Login with `ubuntu` / `armlabs`.

To SSH from another terminal:
```bash
ssh -p 2222 ubuntu@localhost
```

To shut down the VM cleanly: `sudo poweroff` inside the guest.

---

## 5. Install ACfL and ArmPL inside the VM

SSH into the VM and run:

```bash
ssh -p 2222 ubuntu@localhost

# Update packages
sudo apt update && sudo apt install -y build-essential wget environment-modules

# Download ACfL 25.04 (includes ArmPL)
# Get the latest URL from: https://developer.arm.com/Tools%20and%20Software/Arm%20Compiler%20for%20Linux
wget https://developer.arm.com/downloads/-/arm-compiler-for-linux/25-04 -O acfl.tar.gz
tar xf acfl.tar.gz
cd arm-compiler-for-linux_25.04_Ubuntu-24.04
sudo ./install.sh --accept-eula

# Load modules
source /etc/profile.d/modules.sh
module load acfl/25.04
module load armpl/25.04

# Verify
armclang --version
armpl-info
```

**Note:** ACfL is free for use on Arm hardware. The QEMU VM reports `aarch64` to the installer. ArmPL ships bundled with ACfL — no separate download needed.

---

## 6. Set up the Python environment and run labs

```bash
# Inside the VM
sudo apt install -y python3.11 python3.11-venv git

# Clone or copy the project
# Option A: git clone your repo
# Option B: scp from host
#   (from host) scp -P 2222 -r /path/to/ARM_SW_Stack ubuntu@localhost:~/

cd ~/ARM_SW_Stack
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start JupyterLab (accessible from host browser)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Open `http://localhost:8888` in your host browser.

### Lab 1: Replace mock with real ACfL output

```bash
# Inside the VM, with acfl module loaded:
armclang -O3 -mcpu=neoverse-v3 -march=armv9.2-a+sve2 -fvectorize \
    -fopt-info-vec your_code.c 2>&1 | tee code/data/real_acfl_output.txt
```

Then update Lab 1 cell 6 to call `load_vectorization_report(Path('real_acfl_output.txt'))`.

### Lab 2: Replace synthetic data with real ArmPL benchmarks

```python
import ctypes
# Load ArmPL: /opt/arm/armpl_25.04/lib/libblas.so
# Call dgemm() over matrix sizes, measure GFLOP/s
# Replace load_library_comparison() with live ctypes bindings
```

### Run the test suite

```bash
pytest code/tests/ -v --tb=short
# or without pytest:
python code/tests/run_tests_standalone.py
```

---

## 7. Cloud ARM alternatives

Local QEMU works for Labs 1-2. For Labs 3-4 (Slurm clusters, Linaro Forge profiling), cloud ARM instances are more practical.

| Provider | Instance | CPU | Approximate cost | Best for |
|----------|----------|-----|-----------------|----------|
| AWS | c7g.xlarge | Graviton3 (Neoverse V1) | ~$0.14/hr | Labs 1-2 |
| AWS | c8g.xlarge | Graviton4 (Neoverse V2) | ~$0.15/hr | Labs 1-2, closest to V3 |
| Oracle Cloud | A1.Flex | Ampere Altra (Neoverse N1) | Free tier (4 cores, 24 GB) | Labs 1-2, zero cost |
| AWS ParallelCluster | Graviton cluster | Neoverse V1/V2 | Varies | Lab 3 (real Slurm) |
| Azure CycleCloud | Cobalt 100 cluster | Neoverse N2 | Varies | Lab 3 (real Slurm) |

ACfL and ArmPL install identically on cloud instances — use the same steps from Section 5.

**Lab 3 (Slurm):** No substitute for a real multi-node cluster. AWS ParallelCluster or Azure CycleCloud can provision Slurm on Graviton/Cobalt nodes.

**Lab 4 (Linaro Forge MAP):** Requires a commercial Linaro Forge license. A 2-week free trial is available at [linaro.org/forge](https://www.linaroforge.com/free-trial). Must run on a real cluster for meaningful profile data.

---

## 8. Known limitations

- **No Neoverse V3 in QEMU.** The highest available CPU model is `neoverse-v1`. Code compiled with `-mcpu=neoverse-v3` will compile successfully, but runtime use of SVE2 512-bit or SME2 instructions may cause SIGILL if those instructions are not in the V1 feature set.
- **x86 emulation is slow.** On Intel/AMD hosts, expect 5-20x slowdown. Benchmark numbers from QEMU are not representative of real Neoverse performance. Use cloud ARM instances for meaningful benchmarks.
- **No Slurm multi-node.** QEMU default networking is NAT-only. Running a multi-node Slurm cluster inside QEMU requires advanced bridge networking and multiple VMs — not practical. Use a cloud HPC provider instead.
- **Disk I/O overhead.** QEMU virtio disk is significantly slower than native SSD. Large-matrix BLAS benchmarks may be I/O-bound in ways they would not be on real hardware.
- **Apple Silicon HVF limitation.** `-accel hvf` with `-cpu host` exposes the host Apple M-series CPU features, which differ from Neoverse. ACfL will still cross-compile for `-mcpu=neoverse-v3` but the resulting binary targets Neoverse, not Apple Silicon.

---

*Pricing approximate as of April 2026. Check provider pricing pages for current rates.*
