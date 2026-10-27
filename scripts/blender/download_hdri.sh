#!/usr/bin/env bash
# 下载合成数据用的室内 HDRI 环境贴图（Poly Haven，CC0，2k，.hdr）到 dataset/hdri/。
# 已存在的文件会跳过，所以可以重复运行。
#
# 用法：scripts/blender/download_hdri.sh [--dir <目录>]
#   --dir  下载到的目录，默认是包内的 dataset/hdri/
# 有文件下载失败时，退出码为 1。

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
target_dir="$(dirname "$(dirname "${script_dir}")")/dataset/hdri"
base_url="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k"

# 大空间的室内场景：体育馆、大厅、厂房、机库
ids=(
    empty_warehouse_01 old_depot old_bus_depot small_hangar_01 small_hangar_02 hangar_interior
    events_hall_interior crossfit_gym climbing_gym squash_court wrestling_gym gym_01 school_hall
)

while (($# > 0)); do
    case "$1" in
    --dir)
        target_dir=$2
        shift 2
        ;;
    -h | --help)
        sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *)
        echo "unknown option: $1" >&2
        exit 2
        ;;
    esac
done

mkdir -p "${target_dir}"
failed=0
for id in "${ids[@]}"; do
    file="${target_dir}/${id}_2k.hdr"
    if [[ -f "${file}" ]]; then
        echo "ok        ${id}"
        continue
    fi

    echo "download  ${id}"
    partial="${file}.part"
    if ! curl --fail --location --retry 3 --progress-bar --output "${partial}" "${base_url}/${id}_2k.hdr"; then
        echo "failed    ${id}" >&2
        rm -f "${partial}"
        failed=1
        continue
    fi
    mv "${partial}" "${file}"
done

exit "${failed}"
