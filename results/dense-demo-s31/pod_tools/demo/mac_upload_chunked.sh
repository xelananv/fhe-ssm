#!/usr/bin/env bash
# mac_upload_chunked.sh -- RUN ON THE MAC: upload ONE big file as N parallel chunks over separate ssh connections
# (the author's proxy caps ONE stream at ~0.5-1.5 MB/s but scales with connections: 6 streams gave 3x on 2026-09-04),
# reassemble on the pod, verify sha256 end to end, resume-safe (chunks already complete on the pod are skipped).
# usage: SSH_KEY=... PORT=... HOST=... [PAR=8] [CHUNK_MB=64] bash mac_upload_chunked.sh <local file> <remote path>
set -uo pipefail
: "${SSH_KEY:?}"; : "${PORT:?}"; : "${HOST:?}"; : "${PAR:=8}"; : "${CHUNK_MB:=64}"
SRC=$1; DST=$2; [ -s "$SRC" ] || { echo "no such file: $SRC"; exit 2; }
SSHO="ssh -i $SSH_KEY -p $PORT -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=40 -o ServerAliveInterval=30"
SIZE=$(stat -f %z "$SRC"); CH=$(( CHUNK_MB * 1048576 )); N=$(( (SIZE + CH - 1) / CH )); B=$(basename "$DST"); RD="$(dirname "$DST")/.chunks_$B"
SHA=$(shasum -a 256 "$SRC" | cut -c1-64)
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(utc)] $SRC -> root@$HOST:$DST  $SIZE B in $N chunks of $CHUNK_MB MiB, PAR=$PAR, sha $SHA"
# 2026-09-17: skip a file that is already complete on the pod (same size and sha) -- a restarted chain must not re-send it
if [ "$($SSHO root@$HOST "[ -f '$DST' ] && [ \$(stat -c %s '$DST') = $SIZE ] && sha256sum '$DST' | cut -c1-64" 2>/dev/null)" = "$SHA" ]; then echo "[$(utc)] already on the pod with the same sha -- skipped"; exit 0; fi
# 2026-09-19: a chunk counts as done only at its FULL size (truncated leftovers are deleted here and sent again)
LASTW=$(( SIZE - (N - 1) * CH ))
$SSHO root@$HOST "mkdir -p '$RD' && cd '$RD' && rm -f c*.part c*.manual && for f in c[0-9]*; do [ -f \"\$f\" ] || continue; i=\${f#c}; w=$CH; [ \"\$i\" = $((N - 1)) ] && w=$LASTW; if [ \"\$(stat -c %s \"\$f\")\" = \"\$w\" ]; then echo \$i; else rm -f \"\$f\"; fi; done | sort -n" > /tmp/done_chunks.$$ 2>/dev/null
have=$(grep -c . /tmp/done_chunks.$$ 2>/dev/null || echo 0); echo "  chunks already on the pod: $have"
T0=$(date +%s)
# BSD xargs caps the -I replacement at 255 bytes: the per-chunk body lives in a worker script (env carries the rest).
W=$(mktemp /tmp/chunkworker.XXXXXX); cat > "$W" <<'WEOF'
#!/bin/bash
i=$1; off=$(( i * CH )); want=$CH; [ $(( off + want )) -gt "$SIZE" ] && want=$(( SIZE - off ))
for a in 1 2 3; do
  got=$(dd if="$SRC" bs=1048576 skip=$(( i * CHUNK_MB )) count="$CHUNK_MB" 2>/dev/null | $SSHO "root@$HOST" "cat > '$RD/c$i.part' && [ \$(stat -c %s '$RD/c$i.part') = $want ] && mv '$RD/c$i.part' '$RD/c$i' && stat -c %s '$RD/c$i'" 2>/dev/null)   # 2026-09-19: size-checked BEFORE the rename -- a killed sender made cat see EOF and a TRUNCATED chunk was renamed as complete (14 of them cost a sha mismatch)
  [ "$got" = "$want" ] && { echo "  chunk $i ok ($(date -u +%H:%M:%SZ))"; exit 0; }
  echo "  chunk $i retry $a (got ${got:-none}, want $want)"; sleep 5
done; echo "  chunk $i FAILED"; exit 1
WEOF
chmod +x "$W"
export SRC CH CHUNK_MB SIZE SSHO HOST RD
seq 0 $((N - 1)) | grep -vxF -f /tmp/done_chunks.$$ | xargs -P "$PAR" -I{} bash "$W" {}
RC=$?; rm -f /tmp/done_chunks.$$ "$W"
T=$(( $(date +%s) - T0 )); echo "[$(utc)] transfer loop rc=$RC in $T s = $(python3 -c "print(round($SIZE/1e6/max(1,$T),2))") MB/s aggregate (incl. already-present chunks)"
[ "$RC" = 0 ] || { echo "some chunks failed -- re-run (resumes)"; exit 1; }
echo "[$(utc)] assembling on the pod + sha256"
$SSHO root@$HOST "cd '$RD' && ls | grep -c '^c[0-9]*$' | grep -qx $N && cat \$(seq -f 'c%g' 0 $((N - 1))) > '$DST' && sha256sum '$DST' | cut -c1-64" > /tmp/remote_sha.$$ 2>&1
RS=$(tail -1 /tmp/remote_sha.$$); rm -f /tmp/remote_sha.$$
if [ "$RS" = "$SHA" ]; then $SSHO root@$HOST "rm -rf '$RD'"; echo "[$(utc)] OK sha256 verified: $DST"; exit 0; fi
echo "[$(utc)] SHA MISMATCH or assembly failed (remote: $RS) -- chunks kept in $RD for a retry"; exit 1
