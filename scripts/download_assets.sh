#!/usr/bin/env bash
# 下载运行所需、但不放进仓库的文件（网络、M3T 模型等）到包内的 assets/，并逐个校验 SHA256。
# 文件清单在 scripts/assets.sha256。已存在且校验通过的文件会跳过，所以可以重复运行。
#
# 用法：scripts/download_assets.sh [--dir <目录>] [--base-url <地址>]
#   --dir       下载到的目录，默认是包内的 assets/
#   --base-url  文件所在的地址，默认是 GitHub Release assets-v1
# 有文件下载失败或校验不通过时，退出码为 1。

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
target_dir="$(dirname "${script_dir}")/assets"
base_url="https://github.com/Shingura/rmcs_energy_trace/releases/download/assets-v1"

while (($# > 0)); do
    case "$1" in
    --dir)
        target_dir=$2
        shift 2
        ;;
    --base-url)
        base_url=$2
        shift 2
        ;;
    -h | --help)
        sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
while read -r checksum name; do
    [[ -z "${checksum}" || "${checksum}" == \#* ]] && continue
    file="${target_dir}/${name}"
    if [[ -f "${file}" ]] && echo "${checksum}  ${file}" | sha256sum --check --status; then
        echo "ok        ${name}"
        continue
    fi

    echo "download  ${name}"
    partial="${file}.part"
    if ! curl --fail --location --retry 3 --progress-bar --output "${partial}" "${base_url}/${name}"; then
        echo "failed    ${name}" >&2
        rm -f "${partial}"
        failed=1
        continue
    fi
    if ! echo "${checksum}  ${partial}" | sha256sum --check --status; then
        echo "checksum mismatch: ${name}" >&2
        rm -f "${partial}"
        failed=1
        continue
    fi
    mv "${partial}" "${file}"
done <"${script_dir}/assets.sha256"

exit "${failed}"
