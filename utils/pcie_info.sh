#!/usr/bin/env bash
# pcie_info.sh — list PCIe link speed, width, and status for all GPUs on the host
# Usage: sudo ./pcie_info.sh [--retrain]
set -euo pipefail

RETRAIN=0
for arg in "$@"; do
    case "$arg" in
        --retrain) RETRAIN=1 ;;
        *) echo "Usage: $0 [--retrain]" >&2; exit 1 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "Run as root (sudo $0)" >&2
    exit 1
fi

if ! command -v lspci &>/dev/null; then
    echo "lspci not found — install pciutils" >&2
    exit 1
fi

retrain_link() {
    local bdf=$1
    # Retraining briefly drops the PCIe link. If the GPU is currently passed
    # through to a running VM, this will crash or hang the guest. Only retrain
    # when no VM is using the device.
    if lspci -vv -s "$bdf" 2>/dev/null | grep -q "Capabilities: <access denied>"; then
        echo "  ✗  Cannot retrain $bdf — capabilities are not accessible (device is likely bound to vfio-pci and in use by a VM)"
        return
    fi
    echo "  → Retraining PCIe link for $bdf ..."
    # CAP_EXP        — symbolic name for the offset of the PCIe capability structure
    #                  in the device's PCI config space (resolved automatically by setpci)
    # +0x10          — offset within the PCIe capability structure: Link Control register
    # .w             — access width: word (16 bits)
    # 0x0020         — bit 5 = Retrain Link (RW1, self-clearing): triggers the link to
    #                  drop and re-negotiate speed/width. The link trains up to the highest
    #                  speed both sides support (limited by LnkCap and LnkCtl2 target speed).
    setpci -s "$bdf" CAP_EXP+0x10.w=0x0020
    sleep 1

    local lnk_info_after lnksta_after
    lnk_info_after=$(lspci -vv -s "$bdf" 2>/dev/null | grep -i "lnksta" || true)
    lnksta_after=$(echo "$lnk_info_after" | grep -i "LnkSta:" | head -1 | sed 's/.*LnkSta://' | xargs)
    echo "  → LnkSta after retrain : ${lnksta_after:-(not available)}"

    if echo "$lnksta_after" | grep -qi "downgraded"; then
        echo "  ✗  Still downgraded — platform may not support a higher speed"
    else
        echo "  ✓  Link is now running at full speed"
    fi
}

# Find all GPU BDFs (VGA-compatible controllers and 3D controllers)
mapfile -t GPU_BDFS < <(lspci -D | awk '/VGA compatible controller|3D controller|Display controller/ {print $1}')

if [[ ${#GPU_BDFS[@]} -eq 0 ]]; then
    echo "No GPUs found."
    exit 0
fi

echo "Found ${#GPU_BDFS[@]} GPU(s)"
echo

for bdf in "${GPU_BDFS[@]}"; do
    name=$(lspci -D -s "$bdf" | sed 's/^[^ ]* //')
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BDF : $bdf"
    echo "  GPU : $name"

    lnk_info=$(lspci -vv -s "$bdf" 2>/dev/null | grep -i "lnkcap\|lnksta\|lnkctl2" || true)

    if [[ -z "$lnk_info" ]]; then
        echo "  PCIe: (capabilities not readable — vfio-pci or permissions)"
        echo
        continue
    fi

    lnkcap=$(echo "$lnk_info"  | grep -i "LnkCap:"  | head -1 | sed 's/.*LnkCap://' | xargs)
    lnksta=$(echo "$lnk_info"  | grep -i "LnkSta:"  | head -1 | sed 's/.*LnkSta://' | xargs)
    lnkctl2=$(echo "$lnk_info" | grep -i "LnkCtl2:" | head -1 | sed 's/.*LnkCtl2://' | xargs)

    echo "  LnkCap  (max possible) : ${lnkcap:-(not available)}"
    echo "  LnkSta  (negotiated)   : ${lnksta:-(not available)}"
    echo "  LnkCtl2 (target)       : ${lnkctl2:-(not available)}"

    if echo "$lnksta" | grep -qi "downgraded"; then
        cap_speed=$(echo "$lnkcap" | grep -oP 'Speed \K[0-9.]+(?=GT/s)' || true)
        sta_speed=$(echo "$lnksta" | grep -oP 'Speed \K[0-9.]+(?=GT/s)' || true)
        echo
        echo "  ⚠  Link is DOWNGRADED: running at ${sta_speed:-?} GT/s, capable of ${cap_speed:-?} GT/s"

        if [[ $RETRAIN -eq 1 ]]; then
            retrain_link "$bdf"
        else
            echo "     To retrain: sudo $0 --retrain"
        fi
    fi

    echo
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
