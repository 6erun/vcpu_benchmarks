#!/usr/bin/env bash
# /etc/libvirt/hooks/qemu
# Retrain PCIe links for GPU passthrough VMs after QEMU starts.
#
# Problem: QEMU issues a PCIe secondary bus reset when initialising VFIO
# devices. On certain AMD EPYC IODs (buses 60/80/a0/c0) this causes the RTX
# 5090 link to retrain to 2.5 GT/s (Gen 1) instead of 16 GT/s (Gen 4).
# This hook fires after QEMU is up (libvirt "started" event) — the VM BIOS is
# running but the NVIDIA driver has not loaded yet. Retraining here means the
# NVIDIA driver sees Gen 4 at init time, giving ~24 GB/s H2D/D2H bandwidth.
#
# Note: libvirt sends domain XML on stdin for start/started/stopped/release.
# We always read stdin first to avoid SIGPIPE on libvirtd's write end.
#
# Libvirt calls this with arguments:
#   $1 = domain name
#   $2 = operation  (prepare|start|started|stopped|release)
#   $3 = sub-operation

set -uo pipefail

VM_NAME="$1"
OPERATION="$2"

# Always drain stdin — libvirt sends domain XML for start/started/stopped/release
# operations. If we exit without reading it, libvirtd gets SIGPIPE.
DOMAIN_XML=$(cat)

# Only act on "started" (VM is up, before NVIDIA driver loads in guest)
[[ "$OPERATION" == "started" ]] || exit 0

LOG=/var/log/libvirt-pcie-retrain.log
echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}] started — checking PCIe links" >> "$LOG"

# For a GPU at e.g. 0000:61:00.0, its root port is 2 sysfs levels up:
#   /sys/bus/pci/devices/0000:61:00.0 -> .../0000:60:00.0/0000:60:03.1/0000:61:00.0
find_root_port() {
    local gpu_bdf="$1"
    local parent
    parent=$(readlink -f "/sys/bus/pci/devices/${gpu_bdf}/..")
    basename "$parent"
}

# Parse domain XML (from stdin, already read into DOMAIN_XML) to find GPU host
# BDFs from <hostdev type='pci'> <source> blocks.
GPU_BDFS=()
in_source=0
while IFS= read -r line; do
    if echo "$line" | grep -q "<source>"; then
        in_source=1
    fi
    if echo "$line" | grep -q "</source>"; then
        in_source=0
    fi
    if [[ $in_source -eq 1 ]]; then
        if echo "$line" | grep -qE "bus='0x[0-9a-f]+' slot='0x[0-9a-f]+' function='0x[0-9a-f]+'"; then
            bus=$(echo "$line" | grep -oP "bus='0x\K[0-9a-f]+")
            slot=$(echo "$line" | grep -oP "slot='0x\K[0-9a-f]+")
            func=$(echo "$line" | grep -oP "function='0x\K[0-9a-f]+")
            bdf=$(printf "0000:%02x:%02x.%d" "0x${bus}" "0x${slot}" "0x${func}")
            if [[ -e "/sys/bus/pci/devices/${bdf}" ]]; then
                GPU_BDFS+=("$bdf")
            fi
        fi
    fi
done < <(echo "$DOMAIN_XML" | awk '/hostdev.*type=.pci/,/\/hostdev/')

if [[ ${#GPU_BDFS[@]} -eq 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}] no GPU hostdevs found, skipping" >> "$LOG"
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}] GPU BDFs: ${GPU_BDFS[*]}" >> "$LOG"

RETRAINED=()
for gpu_bdf in "${GPU_BDFS[@]}"; do
    root_port=$(find_root_port "$gpu_bdf") || continue

    lnk_info=$(lspci -vv -s "$root_port" 2>/dev/null | grep -i "LnkSta:" | head -1 || true)
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}]   ${root_port}: ${lnk_info}" >> "$LOG"

    if echo "$lnk_info" | grep -q "2\.5GT"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}]   retraining ${root_port} (link at 2.5 GT/s)" >> "$LOG"
        # Set root port target speed to Gen 5 (hardware caps at Gen 4 on EPYC Rome)
        setpci -s "$root_port" CAP_EXP+0x30.w=0x0005
        # Retrain Link bit (bit 5 of LnkCtl, self-clearing)
        setpci -s "$root_port" CAP_EXP+0x10.w=0x0020
        sleep 1
        lnk_after=$(lspci -vv -s "$root_port" 2>/dev/null | grep -i "LnkSta:" | head -1 | xargs || true)
        echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}]   after retrain: ${lnk_after}" >> "$LOG"
        RETRAINED+=("${root_port}")
    fi
done

if [[ ${#RETRAINED[@]} -gt 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}] retrained ${#RETRAINED[@]} link(s): ${RETRAINED[*]}" >> "$LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${VM_NAME}] all links already at expected speed" >> "$LOG"
fi

exit 0
