#!/usr/bin/env bash
set -euo pipefail

# 不拉 Docker Hub 镜像，按国内代理链下载官方二进制。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version="${GITLEAKS_VERSION:-8.24.2}"
bindir="${1:-$repo_root/.ci-bin}"
mkdir -p "$bindir"

if [[ -x "$bindir/gitleaks" ]]; then
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
src="https://github.com/gitleaks/gitleaks/releases/download/v${version}/gitleaks_${version}_linux_x64.tar.gz"

normalize_prefix() {
  local prefix="$1"
  [[ -z "$prefix" ]] && return 0
  printf '%s/' "${prefix%/}"
}

try_get() {
  curl -fL --retry 2 --retry-delay 2 --connect-timeout 20 --max-time 180 -o "$tmp/gitleaks.tgz" "$1"
}

prefixes=()
if [[ -n "${GITHUB_PROXY:-}" ]]; then
  prefixes+=("$(normalize_prefix "$GITHUB_PROXY")")
fi
prefixes+=(
  "https://ghfast.top/"
  "https://gh-proxy.com/"
  ""
)

seen=" "
ok=0
for prefix in "${prefixes[@]}"; do
  case "$seen" in
    *" ${prefix} "*) continue ;;
  esac
  seen+="${prefix} "
  if [[ -z "$prefix" ]]; then
    url="$src"
  else
    url="${prefix}${src}"
  fi
  echo "fetch: $url"
  if try_get "$url"; then
    ok=1
    break
  fi
done

if [[ "$ok" -ne 1 ]]; then
  echo "failed to download gitleaks ${version}" >&2
  exit 1
fi

tar -xzf "$tmp/gitleaks.tgz" -C "$tmp"
install -m 0755 "$tmp/gitleaks" "$bindir/gitleaks"
