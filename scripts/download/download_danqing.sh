#!/bin/bash
# ============================================================
# Download ~100,000 samples from DanQing (ModelScope)
#
# Dataset: deepglint/DanQing
# Total files : 33,298 parquet shards
# Est. rows/file: ~3,003  (100M total / 33,298 files)
# Files to fetch: 34  →  ~102,000 samples
# ============================================================

DATASET_ID="deepglint/DanQing"
LOCAL_DIR="./data/DanQing"
TOTAL_SHARDS=33298
NUM_FILES=35              # 34 × ~3,003 ≈ 102,000 samples for training and additional 1 file for testing
SEED=42                   # change for a different random draw
MAX_JOBS=6                # parallel downloads (tune to your bandwidth)
MAX_RETRIES=3             # attempts per shard before giving up and substituting
RETRY_DELAY=5             # seconds between retries

mkdir -p "$LOCAL_DIR"

# Clean up any leftover temp dirs if the script itself is interrupted or killed
trap 'echo ""; echo "Interrupted — cleaning up temp dirs..."; rm -rf "${LOCAL_DIR}"/.tmp_*; exit 1' INT TERM

echo "Randomly selecting $NUM_FILES primary + $(( NUM_FILES * 2 )) candidate shards (seed=$SEED) ..."

# Generate primary pool (NUM_FILES) + candidate pool (2×NUM_FILES) in one awk pass.
# Fisher-Yates on the first NUM_FILES*3 positions gives us reproducible unique indices.
POOL_SIZE=$(( NUM_FILES * 3 ))
ALL_INDICES=$(awk -v n="$TOTAL_SHARDS" -v k="$POOL_SIZE" -v seed="$SEED" '
BEGIN {
    srand(seed)
    for (i = 0; i < n; i++) a[i] = i
    limit = (k < n) ? k : n
    for (i = 0; i < limit; i++) {
        j = int(rand() * (n - i)) + i
        tmp = a[i]; a[i] = a[j]; a[j] = tmp
    }
    for (i = 0; i < limit; i++) print a[i]
}')

SELECTED_INDICES=$(echo "$ALL_INDICES" | head -n "$NUM_FILES" | sort -n)
# Candidates: everything after the first NUM_FILES rows (unsorted — tried in shuffle order)
CANDIDATE_INDICES=$(echo "$ALL_INDICES" | awk "NR > $NUM_FILES")

if [ -z "$SELECTED_INDICES" ]; then
    echo "ERROR: awk produced no output."
    exit 1
fi

echo "Primary shard indices:"
echo "$SELECTED_INDICES"
echo ""

# ── Worker function ─────────────────────────────────────────────────────────
# Retries up to MAX_RETRIES times. On each attempt it uses a fresh temp dir
# (old one is cleaned up before the next try). Returns 0 on success, 1 if all
# attempts are exhausted.
download_shard() {
    local label="$1"   # e.g. "[3/34]" or "[sub-2]"
    local idx="$2"
    local padded
    padded=$(printf "%05d" "$idx")
    local filename="train-${padded}-of-${TOTAL_SHARDS}.parquet"
    local dest_dir="${LOCAL_DIR}/data"
    local dest_file="${dest_dir}/${filename}"

    # Skip if already fully downloaded (real shard > 100 MB)
    local size
    size=$(stat -c%s "$dest_file" 2>/dev/null || stat -f%z "$dest_file" 2>/dev/null || echo 0)
    if [ "$size" -gt 104857600 ]; then
        echo "$label Already exists, skipping: $filename"
        return 0
    fi

    local attempt tmp_dir
    for attempt in $(seq 1 "$MAX_RETRIES"); do
        tmp_dir=$(mktemp -d "${LOCAL_DIR}/.tmp_${idx}_XXXXXX")

        if [ "$attempt" -eq 1 ]; then
            echo "$label Downloading $filename ..."
        else
            echo "$label Retry $((attempt - 1))/$((MAX_RETRIES - 1)) for $filename (${RETRY_DELAY}s delay) ..."
            sleep "$RETRY_DELAY"
        fi

        # MODELSCOPE_LOG_LEVEL=40 (logging.ERROR) suppresses SDK logger noise.
        # stdout (ASCII-art banner) goes to /dev/null; tqdm progress stays on stderr.
        MODELSCOPE_LOG_LEVEL=40 modelscope download \
            --dataset "$DATASET_ID" \
            --local_dir "$tmp_dir" \
            --include "data/${filename}" \
            >/dev/null

        local downloaded_file="${tmp_dir}/data/${filename}"
        if [ -f "$downloaded_file" ]; then
            mkdir -p "$dest_dir"
            mv "$downloaded_file" "$dest_file"
            rm -rf "$tmp_dir"
            echo "  ✓ $filename saved"
            return 0
        else
            echo "  ✗ Attempt $attempt/$MAX_RETRIES failed for $filename"
            rm -rf "$tmp_dir"
        fi
    done

    echo "  ✗ All $MAX_RETRIES attempts failed for $filename — will try a substitute"
    return 1
}
# ────────────────────────────────────────────────────────────────────────────

# ── Phase 1: parallel primary downloads ─────────────────────────────────────
COUNTER=0
TOTAL=$(echo "$SELECTED_INDICES" | wc -w)
JOBS=0

for idx in $SELECTED_INDICES; do
    COUNTER=$((COUNTER + 1))

    download_shard "[$COUNTER/$TOTAL]" "$idx" &

    JOBS=$((JOBS + 1))
    if [ "$JOBS" -ge "$MAX_JOBS" ]; then
        wait -n 2>/dev/null || wait
        JOBS=$((JOBS - 1))
    fi
done

# Wait for all remaining background jobs
wait

# ── Phase 2: fill up to NUM_FILES if any primaries failed ───────────────────
DOWNLOADED=$(ls "$LOCAL_DIR"/data/*.parquet 2>/dev/null | wc -l)
STILL_NEEDED=$(( NUM_FILES - DOWNLOADED ))

if [ "$STILL_NEEDED" -gt 0 ]; then
    echo ""
    echo "⚠  $STILL_NEEDED primary shard(s) failed. Trying substitute shards ..."
    SUB_COUNTER=0
    for idx in $CANDIDATE_INDICES; do
        [ "$STILL_NEEDED" -le 0 ] && break
        SUB_COUNTER=$((SUB_COUNTER + 1))
        download_shard "[sub-$SUB_COUNTER]" "$idx"
        if [ $? -eq 0 ]; then
            STILL_NEEDED=$(( STILL_NEEDED - 1 ))
        fi
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
DOWNLOADED=$(ls "$LOCAL_DIR"/data/*.parquet 2>/dev/null | wc -l)
echo "Done. Downloaded $DOWNLOADED / $NUM_FILES parquet files."
if [ "$DOWNLOADED" -lt "$NUM_FILES" ]; then
    echo "⚠  $(( NUM_FILES - DOWNLOADED )) file(s) could not be downloaded after all substitutes exhausted."
fi
echo "Files saved to: $LOCAL_DIR/data/"
