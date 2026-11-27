#!/usr/bin/env bash
set -uo pipefail
master=${DLOAN_MASTER:?} host=${DLOAN_HOST:?} admin=${DLOAN_ADMIN:-} idx=${DLOAN_INDEX:-0}
gb=${DLOAN_GB:-600} nseg=${DLOAN_SPLIT:-2} period=${DLOAN_PERIOD:-10} thr=${DLOAN_THREADS:-4} port0=${DLOAN_PORT:-50052}
out=${DLOAN_OUT:-/logs/dloan}; mkdir -p "$out"
hn=$(hostname -s)
nics=("${DLOAN_NICS0:-mlx5_0,mlx5_1,mlx5_2,mlx5_5}" "${DLOAN_NICS1:-mlx5_6,mlx5_7,mlx5_8,mlx5_11}")   # DLOAN_NICS<k>: per-client NIC list override
cpus=("64-79,240-255" "120-135,296-311")
allnics="${nics[0]},${nics[1]}"
log() { echo "[dloan $hn $(date '+%F %T')] $*"; }
export MC_SLICE_SIZE=${MC_SLICE_SIZE:-1048576} MC_MAX_MR_SIZE=${MC_MAX_MR_SIZE:-4294967296} WITH_NVIDIA_PEERMEM=0
pids=()
cleanup() { log "stopping clients ${pids[*]}"; kill "${pids[@]}" 2>/dev/null; wait; exit 0; }
trap cleanup TERM INT
log "cpus allowed: $(grep Cpus_allowed_list /proc/self/status | cut -f2); master $master admin ${admin:-none}; ${gb} GB in $nseg segment(s)"
taskset -cp "${cpus[0]},${cpus[1]}" $$ > /dev/null 2>&1 && log "monitor pinned to ${cpus[0]},${cpus[1]}"
seg=$((gb / nseg))
for ((k = 0; k < nseg; k++)); do
    dev=$allnics bind=()
    if ((nseg == 2)); then
        dev=${nics[k]}
        if numactl --physcpubind="${cpus[k]}" --membind="$k" true 2> /dev/null; then
            bind=(numactl --physcpubind="${cpus[k]}" --membind="$k")
        else
            log "physcpubind ${cpus[k]} refused in this step; binding memory only"
            bind=(numactl --membind="$k")
        fi
    fi
    "${bind[@]}" mooncake_client --master_server_address="$master" --metadata_server=P2PHANDSHAKE \
        --global_segment_size="${seg}GB" --local_buffer_size=1GB --protocol=rdma --device_names="$dev" \
        --host="$host" --port=$((port0 + k)) --threads="$thr" --logtostderr \
        > "$out/client-$hn-n$k.log" 2>&1 &
    pids+=($!)
    log "client $k pid $! (${bind[*]:-no binding}) ${seg}GB devices $dev port $((port0 + k))"
done
ready=0
for ((t = 0; t < 900; t += 5)); do
    for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null || { log "client $p exited before mounting"; tail -20 "$out"/client-"$hn"-n*.log; exit 1; }; done
    if grep -q "Failed to mount segment\|MountSegmentAndGetId failed" "$out"/client-"$hn"-n*.log; then
        log "mount failed"; tail -20 "$out"/client-"$hn"-n*.log; exit 1
    fi
    if [[ -n $admin ]]; then
        n=$(curl -s -m 5 "http://$admin/get_all_segments" | grep -o "$host:[0-9]*" | sort -u | wc -l)
        ((n >= nseg)) && { log "master lists $n segment(s) for $host"; ready=1; break; }
    fi
    if ((t >= 60)) && (($(grep -l "Mounting segment" "$out"/client-"$hn"-n*.log 2> /dev/null | wc -l) >= nseg)); then
        log "clients alive with 'Mounting segment' after ${t}s (master listing: ${n:-n/a})"; ready=1; break
    fi
    sleep 5
done
((ready)) || { log "not mounted after 900 s"; exit 1; }
echo "DLOAN-READY $hn $(date +%s)"
csv=$out/mon-$hn.csv
hdr="ts"
for f in MemAvailable MemFree AnonPages Shmem Mlocked; do hdr+=",$f"; done
for nn in 0 1; do hdr+=",n${nn}_MemFree,n${nn}_FilePages"; done
hdr+=",client_rss_kb"
ibs=$(ls /sys/class/infiniband 2> /dev/null | grep -E '^mlx5_(0|1|2|5|6|7|8|11)$' | sort -V)
for d in $ibs; do hdr+=",${d}_rcv,${d}_xmit"; done
echo "$hdr" > "$csv"
while :; do
    row=$(date +%s.%N | cut -c1-14)
    for f in MemAvailable MemFree AnonPages Shmem Mlocked; do row+=",$(awk -v k="$f:" '$1 == k {print $2}' /proc/meminfo)"; done
    for nn in 0 1; do
        m=/sys/devices/system/node/node$nn/meminfo
        row+=",$(awk '$3 == "MemFree:" {print $4}' "$m" 2> /dev/null),$(awk '$3 == "FilePages:" {print $4}' "$m" 2> /dev/null)"
    done
    rss=0; for p in "${pids[@]}"; do r=$(awk '$1 == "VmRSS:" {print $2}' "/proc/$p/status" 2> /dev/null); rss=$((rss + ${r:-0})); done
    row+=",$rss"
    for d in $ibs; do
        c=/sys/class/infiniband/$d/ports/1/counters
        row+=",$(($(cat $c/port_rcv_data 2> /dev/null || echo 0) * 4)),$(($(cat $c/port_xmit_data 2> /dev/null || echo 0) * 4))"
    done
    echo "$row" >> "$csv"
    if [[ $idx == 0 && -n $admin ]]; then
        seg=$(curl -s -m 5 "http://$admin/get_segments_detail" | tr -d '\n')
        [[ $seg == "{"* ]] && printf '{"ts": %s, "segments": %s}\n' "$(date +%s)" "$seg" >> "$out/master.jsonl"
    fi
    for p in "${pids[@]}"; do kill -0 "$p" 2> /dev/null || log "WARNING client $p exited"; done
    sleep "$period" & wait $!
done
