#!/bin/bash

# 800 Picacho Lane - HDR Production Pipeline
# ==========================================

search_and_select() {
    local prompt="$1" exts="$2" query
    local files choice pattern
    while true; do
        read -rp "$prompt" query
        [ -z "$query" ] && return 1
        pattern=".*\.(${exts})"
        IFS=$'\n' read -r -d '' -a files < <(find . -type f -iname "*${query}*" -iregex "$pattern" -print0)
        if [ ${#files[@]} -eq 0 ]; then
            echo "No matches found for '$query'."
            continue
        fi
        if [ ${#files[@]} -gt 1 ]; then
            echo "Select a file:" >&2
            select choice in "${files[@]}" "Search again"; do
                if [ "$REPLY" -eq $((${#files[@]}+1)) ]; then
                    choice=""
                    break
                elif [ -n "$choice" ]; then
                    echo "$choice"
                    return 0
                else
                    echo "Invalid selection." >&2
                fi
            done
            [ -n "$choice" ] && return 0
        else
            echo "${files[0]}"
            return 0
        fi
    done
}

while true; do
    INPUT=$(search_and_select "Search term for input video (or press Enter to finish): " "mp4|mov")
    [ -z "$INPUT" ] && break
    LUT_PATH=$(search_and_select "Search term for 3D LUT: " "cube") || continue

    LAT=$(exiftool -GPSLatitude# -s3 "$INPUT" 2>/dev/null)
    LON=$(exiftool -GPSLongitude# -s3 "$INPUT" 2>/dev/null)
    if [ -n "$LAT" ] && [ -n "$LON" ]; then
        COORD_SUFFIX="_${LAT}_${LON}"
    else
        COORD_SUFFIX=""
    fi

    OUTPUT_DIR="HDR_Production_$(date +%Y%m%d_%H%M%S)${COORD_SUFFIX}"
    mkdir -p "$OUTPUT_DIR"

    BASENAME=$(basename "${INPUT%.*}")

    echo "📊 Phase 1: Creating HDR Master..."
    ffmpeg -i "$INPUT" \
      -vf "format=p010le,\
           lut3d='$LUT_PATH',\
           zscale=t=linear:npl=100,\
           zscale=t=smpte2084:p=bt2020:m=bt2020nc:r=tv,\
           format=yuv420p10le" \
      -c:v hevc_videotoolbox -profile:v main10 -b:v 50M \
      -color_primaries 9 -color_trc 16 -colorspace 9 \
      -metadata:s:v "title=800 Picacho Lane - $(basename "$LUT_PATH" .cube)" \
      -c:a copy \
      "$OUTPUT_DIR/${BASENAME}_HDR_Master.mov"

    echo "🌐 Phase 2: Creating Web Deliverables..."
    ffmpeg -i "$OUTPUT_DIR/${BASENAME}_HDR_Master.mov" \
      -c:v libx265 -preset slow -crf 18 \
      -x265-params "hdr-opt=1:repeat-headers=1" \
      -c:a aac -b:a 192k \
      "$OUTPUT_DIR/${BASENAME}_YouTube_HDR.mp4"

    ffmpeg -i "$OUTPUT_DIR/${BASENAME}_HDR_Master.mov" \
      -vf "zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:p=bt709:m=bt709" \
      -c:v h264_videotoolbox -b:v 20M \
      "$OUTPUT_DIR/${BASENAME}_MLS_Premium.mp4"

    echo "✨ Production complete! Files in: $OUTPUT_DIR"
done
