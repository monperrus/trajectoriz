#!/bin/sh
# Release bricked FUSE mounts by aborting their kernel connections.
#
# When a FUSE daemon stops answering, the kernel keeps holding the mount's
# inode lock and every process that touches the directory ends up in
# uninterruptible sleep — D state, which SIGKILL cannot end. Killing the
# daemon does not help: one of its own threads is usually stuck in the same
# way. Unmounting does not help either — `fusermount -uz` detaches the
# directory but leaves the pending requests pending.
#
# The only way out is to abort the connection, which makes every request in
# flight return an error. Then the daemon dies, and its callers are reaped.
#
# Order: abort, then kill the daemons, then unmount the leftovers. This script
# does all three (the last two only for trajectoriz MemoryFS mounts, which it
# owns; other daemons are left to their supervisors).
#
# Usage:
#   sh scripts/fuse-abort.sh            # dry run: print what would be aborted
#   sudo sh scripts/fuse-abort.sh --go  # do it
#
# KEEP is the whole safety mechanism: connections serving a mount whose source
# or fstype matches it are never touched. Aborting a healthy connection breaks
# whatever uses it — a flatpak document portal, a running AppImage's bundle, a
# mounted remote. Read the dry run and make sure everything you recognise is
# in the "keeping" line before passing --go.
#
# Root is only needed for mounts belonging to another user: the abort files are
# writable by the user who mounted them. Note that `sudo echo 1 > …` redirects
# as you, not as root — hence the `echo 1 > "$c/abort"` inside this script.
set -u

KEEP=${KEEP:-'portal|gvfs|AppImage|sshfs|rclone|encfs|gocryptfs|s3fs|curlftpfs'}
CONNECTIONS=${CONNECTIONS:-/sys/fs/fuse/connections}
MOUNTINFO=${MOUNTINFO:-/proc/self/mountinfo}
GO=${1:-}

if [ ! -d "$CONNECTIONS" ] || [ -z "$(ls -A "$CONNECTIONS" 2>/dev/null)" ]; then
    echo "No FUSE connections listed in $CONNECTIONS." >&2
    echo "If it is empty, mount fusectl: sudo mount -t fusectl none $CONNECTIONS" >&2
    exit 1
fi

# A FUSE connection is identified by the minor number of its device, which
# mountinfo gives as field 3. Everything here reads /proc and /sys only: a
# stat() of a wedged mountpoint would hang, and diagnosing the hang must not
# be what hangs.
_fuse_rows() {
    awk '{sep = 0; for (i = 7; i <= NF; i++) if ($i == "-") { sep = i; break }}
         sep && ($(sep + 1) == "fuse" || $(sep + 1) ~ /^fuse\./) {
             split($3, d, ":")
             print d[2] "\t" $(sep + 1) "\t" $(sep + 2) "\t" $5
         }' "$MOUNTINFO"
}

rows=$(_fuse_rows)
keep_ids=$(printf '%s\n' "$rows" | awk -F'\t' -v keep="$KEEP" \
    '$2 ~ keep || $3 ~ keep { print $1 }')

echo "FUSE mounts:"
if [ -n "$rows" ]; then
    printf '%s\n' "$rows" | while IFS='	' read -r id fstype source point; do
        printf '  %-5s %-12s %-28s %s\n' "$id" "$fstype" "$source" "$point"
    done
else
    echo "  (none — every connection below is orphaned)"
fi
echo "Keeping connections: $(printf '%s' "${keep_ids:-none}" | tr '\n' ' ')"
echo

aborted=0
for c in "$CONNECTIONS"/*; do
    id=${c##*/}
    skip=0
    for k in $keep_ids; do
        [ "$id" = "$k" ] && skip=1
    done
    [ "$skip" = 1 ] && continue

    waiting=$(cat "$c/waiting" 2>/dev/null || echo "?")
    point=$(printf '%s\n' "$rows" | awk -F'\t' -v i="$id" '$1 == i { print $4 }')
    [ -n "$point" ] || point="(orphaned: no mountinfo row)"

    if [ "$GO" = "--go" ]; then
        if echo 1 > "$c/abort" 2>/dev/null; then
            echo "aborted $id  waiting=$waiting  $point"
            aborted=$((aborted + 1))
        else
            echo "FAILED to abort $id ($point) — try as root" >&2
        fi
    else
        echo "would abort $id  waiting=$waiting  $point"
    fi
done

if [ "$GO" != "--go" ]; then
    echo
    echo "Dry run. Re-run with --go to abort the connections listed above."
    exit 0
fi

echo
echo "Aborted $aborted connection(s). Reaping trajectoriz memory daemons:"
pkill -f 'trajectoriz-cli memory' 2>/dev/null && echo "  signalled trajectoriz-cli memory"
pkill -f 'trajectoriz\.cli memory' 2>/dev/null && echo "  signalled python -m trajectoriz.cli memory"
sleep 1

echo "Detaching leftover MemoryFS mountpoints:"
_fuse_rows | awk -F'\t' '$3 == "MemoryFS" { print $4 }' | while read -r mp; do
    if fusermount -uz "$mp" 2>/dev/null || umount -l "$mp" 2>/dev/null; then
        echo "  detached $mp"
    else
        echo "  could not detach $mp — it may already be gone" >&2
    fi
done

echo
echo "Remaining D-state processes (should be empty, kernel threads aside):"
ps -eo pid,stat,wchan:20,cmd | awk '$2 ~ /D/ && $1 != "PID"' || true
