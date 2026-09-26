#!/usr/bin/env bash
# mac_upload_keys.sh -- RUN ON THE MAC: upload the Mac-minted EVAL material (never secret.key) to the pod,
#                       PAR files at a time (the author's proxy capped ONE stream at ~1.5 MB/s on 2026-09-04 while
#                       6 streams gave ~4.7 MB/s aggregate), resumable (rsync --partial), then verify the
#                       MANIFEST on the pod and assert the secret is absent there.
# usage:  SSH_KEY=~/.ssh/<pod-key> PORT=... HOST=... [SRC=~/mac_keys_r17] [DEST=/root/mac_keys_r17] [PAR=4] \
#           bash mac_upload_keys.sh
# The key set: 36 rotation-key parts +
# evalmult 352 MB + public.key 88 MB + cryptocontext.bin(+.dev) + contract_r17.json + MANIFEST.sha256.
set -uo pipefail
: "${SSH_KEY:?}"; : "${PORT:?}"; : "${HOST:?}"
: "${SRC:=$HOME/mac_keys_r17}"; : "${DEST:=/root/mac_keys_r17}"; : "${PAR:=4}"
SSHO="ssh -i $SSH_KEY -p $PORT -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-keyup-%r@%h:%p -o ControlPersist=900 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
[ -s "$SRC/secret.key" ] || { echo "no secret.key in $SRC -- is this the key dir?"; exit 2; }
[ -s "$SRC/MANIFEST.sha256" ] || { echo "no MANIFEST.sha256 in $SRC"; exit 2; }
LIST=$(mktemp)
( cd "$SRC" && ls -1 | grep -vE '^(secret\.key|keygen\.log|keygen\.stderr)$' | grep -v '^\.' ) | awk '{ if ($0 ~ /evalrot/) print "1 " $0; else print "0 " $0 }' | sort | cut -d' ' -f2- > "$LIST"
N=$(grep -c . "$LIST"); TOT=$(cd "$SRC" && cat "$LIST" | xargs stat -f %z 2>/dev/null | awk '{s+=$1} END{print s}')
echo "[$(utc)] $N files, $(python3 -c "print(round($TOT/1e9,2))") GB from $SRC -> root@$HOST:$DEST (PAR=$PAR). secret.key EXCLUDED."
grep -q secret "$LIST" && { echo "REFUSED: secret in the file list"; exit 3; }
$SSHO "root@$HOST" "mkdir -p $DEST && test ! -e $DEST/secret.key && echo 'dest ok, no secret there'" || exit 3
T0=$(date +%s)
# BSD xargs caps the -I replacement at 255 bytes: per-file body in a worker script, env carries the rest.
W=$(mktemp /tmp/keyworker.XXXXXX); cat > "$W" <<'WEOF'
#!/bin/bash
f=$1
for a in 1 2 3; do
  rsync -a --partial --partial-dir=.rsync-partial -e "$SSHO" "$SRC/$f" "root@$HOST:$DEST/$f" && { echo "  done $f ($(date -u +%H:%M:%SZ))"; exit 0; }
  echo "  retry $a $f"; sleep 10
done; echo "  FAILED $f"; exit 1
WEOF
chmod +x "$W"; export SSHO SRC HOST DEST
cat "$LIST" | xargs -P "$PAR" -I{} bash "$W" {}
rm -f "$W"
echo "[$(utc)] transfer loop finished in $(( $(date +%s) - T0 )) s ($(python3 -c "print(round($TOT/1e6/max(1,$(date +%s)-$T0),2))") MB/s aggregate)"
echo "[$(utc)] verifying on the pod (sha256sum -c MANIFEST.sha256; secret.key must be absent)"
$SSHO "root@$HOST" "cd $DEST && test ! -e secret.key && sha256sum -c --quiet MANIFEST.sha256 && ls -1 | wc -l && du -sh . && echo KEYUP_OK" || { echo "VERIFY FAILED -- re-run (rsync resumes)"; exit 1; }
rm -f "$LIST"
