#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rmnp_dir="${1:-${package_root}/RMNP}"
rmnp_commit="a662ebcfc5a76c23f6ddee7a0d9d4b883c847478"

if [[ ! -d "${rmnp_dir}/.git" ]]; then
  git clone https://github.com/Dominator-Index/RMNP.git "${rmnp_dir}"
fi

git -C "${rmnp_dir}" fetch origin
git -C "${rmnp_dir}" checkout --detach "${rmnp_commit}"
cp -a "${package_root}/rmnp_overlay/." "${rmnp_dir}/"

echo "RMNP prepared at ${rmnp_dir}"
echo "Pinned upstream commit: ${rmnp_commit}"
echo "Run: python ${package_root}/campaign/controller.py --check"
