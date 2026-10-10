#!/usr/bin/env bash
# Update beellama4.cpp (BeeLlama v0.4.8): pull from upstream, optionally
# cherry-pick PRs, apply the GCC-15 patch, configure, build llama-server only.
# Usage: ./update.sh [PR_NUMBER ...]
#   CLI args are appended to the PRS list below
#
# This folder is the dedicated v0.4.8 source tree (backend tag: bee4).
# The 3.x tree (engines/beellama.cpp) was retired 2026-10-01 to /home/guy/serve/archive/.
set -euo pipefail

cd "$(dirname "$0")"

# Upstream branch to track. v0.4.8 is the dev branch (ggml 0.26.0, llama.cpp
# master sync through 9adc7f420); the latest release TAG is v0.4.7. Bump this
# when a newer dev branch or release tag appears.
UPSTREAM_BRANCH="v0.4.8"

# PRs to cherry-pick on every build. Edit this list as needed.
PRS=()
#PRS+=(22673)

# Append any CLI args
PRS+=("$@")

echo "=== Fetching upstream ==="
# Discard any locally-applied patches so rebase has a clean tree (re-applied below).
git checkout -- ggml/src/ggml-cpu/ops.cpp 2>/dev/null || true
git checkout -- ggml/src/ggml-common.h 2>/dev/null || true
# Fetch first so the remote-tracking ref exists for the checkout. An upstream
# release TAG of the same name would otherwise win resolution and detach HEAD;
# create the local branch explicitly instead of relying on checkout's DWIM.
# Upstream has re-tagged some old preview tags, and "would clobber existing tag"
# rejections make git fetch exit 1 even though branch refs update fine. The
# show-ref checks below verify the branch ref actually arrived.
git fetch upstream --tags || true
if git show-ref --verify --quiet "refs/remotes/upstream/${UPSTREAM_BRANCH}"; then
    UPSTREAM_REF="upstream/${UPSTREAM_BRANCH}"
elif git show-ref --verify --quiet "refs/tags/${UPSTREAM_BRANCH}"; then
    UPSTREAM_REF="tags/${UPSTREAM_BRANCH}"
else
    echo "!!! upstream has neither branch nor tag '${UPSTREAM_BRANCH}'"
    exit 1
fi
if git show-ref --verify --quiet "refs/heads/${UPSTREAM_BRANCH}"; then
    git checkout "$UPSTREAM_BRANCH"
else
    git checkout -b "$UPSTREAM_BRANCH" "$UPSTREAM_REF"
fi

# Check how far behind we are
BEHIND=$(git rev-list --count "HEAD..${UPSTREAM_REF}" || echo 0)
echo "  Upstream ${UPSTREAM_REF} is ${BEHIND} commits ahead of HEAD"

# Rebase onto upstream to get clean history
git rebase "$UPSTREAM_REF"

# Push updated branch to origin (use refs/heads to avoid tag/branch refspec
# ambiguity). Non-fatal: the build proceeds from local HEAD regardless.
git push origin "refs/heads/${UPSTREAM_BRANCH}" --force-with-lease \
    || echo "  (push to origin failed/skipped — non-fatal)"

# Cherry-pick PRs by number (fetches from origin and cherry-picks the merge commit)
if [ ${#PRS[@]} -gt 0 ]; then
    echo ""
    echo "=== Cherry-picking PRs: ${PRS[*]} ==="
    for pr in "${PRS[@]}"; do
        echo "--- Fetching PR #${pr} ---"
        git fetch origin "refs/pull/${pr}/merge:pr-${pr}"
        echo "--- Cherry-picking PR #${pr} ---"
        git cherry-pick --no-commit -m 1 "pr-${pr}" || {
            echo "!!! Cherry-pick of PR #${pr} failed, aborting"
            git cherry-pick --abort 2>/dev/null || true
            exit 1
        }
    done
    # Commit cherry-picks
    git commit -m "cherry-pick PRs: ${PRS[*]}"
    git push origin "refs/heads/${UPSTREAM_BRANCH}" --force-with-lease \
        || echo "  (push to origin failed/skipped — non-fatal)"
fi

echo ""
echo "=== Applying GCC-15 patch (idempotent) ==="
OPS="ggml/src/ggml-cpu/ops.cpp"
if grep -q '#include <cstring>' "$OPS"; then
    echo "  already patched ($OPS has #include <cstring>)"
else
    # GCC 15 std::memcpy/std::memset need <cstring>; insert after <cmath>.
    sed -i '/#include <cmath>/a #include <cstring>' "$OPS"
    echo "  patched $OPS (+ #include <cstring>)"
fi

echo ""
echo "=== Applying stored local patches (idempotent) ==="
# patches/*.patch hold local fixes for dev-branch breakage (e.g. the v0.4.8
# _MSC_VER guard around the PQ2_0/PTQ1_0 typedefs in ggml-common.h). A patch
# that no longer applies is skipped — if upstream already fixed it, the build
# will prove it.
for p in patches/*.patch; do
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"
        echo "  applied $p"
    else
        echo "  skipped (already applied or not applicable): $p"
    fi
done

echo ""
echo "=== Configuring with CUDA ==="

# Find CUDA toolkit (prefer 13.x for Blackwell support)
CUDA_PATH=$(ls -d /usr/local/cuda-13.* /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1)
if [ -z "$CUDA_PATH" ]; then
    echo "!!! CUDA toolkit not found. Install CUDA 12.x or 13.x."
    exit 1
fi

CUDA_COMPILER="$CUDA_PATH/bin/nvcc"
if [ ! -x "$CUDA_COMPILER" ]; then
    echo "!!! CUDA compiler not found at $CUDA_COMPILER"
    exit 1
fi

echo "Using CUDA toolkit: $CUDA_PATH"

# Build for both RTX 3090 (sm_86) and RTX 5090 (sm_120a)
# Clear CMake cache to ensure architecture detection is fresh
rm -f build/CMakeCache.txt

cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="86;120a" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "=== Building llama-server (target only) ==="
cmake --build build --target llama-server --config Release -j "$(nproc)"

echo ""
echo "=== Done ==="
"$PWD/build/bin/llama-server" --version 2>/dev/null | head -2 || true
