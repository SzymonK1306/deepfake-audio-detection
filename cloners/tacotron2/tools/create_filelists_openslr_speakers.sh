#!/bin/bash

speaker_info_file=$1
dataset_dir=$2

echo "Debugging info:"
echo "Speaker info file: $speaker_info_file"
echo "Dataset directory: $dataset_dir"
echo ""

if [ -z "$speaker_info_file" ] || [ -z "$dataset_dir" ]; then
    echo "Usage: $0 <speaker_info_file> <dataset_dir>"
    exit 1
fi

current_dir=$(pwd)
filelist_dir="filelists"
mkdir -p "$filelist_dir"

process_trans_file() {
    local trans_file=$1
    local dir_path=$(dirname "$trans_file")
    local subset=$2
    local speaker_id=$3
    local gender=$4
    local minutes=$5
    local name=$6

    local formatted_name=$(echo "$name" | tr ' ' '-')
    local base_name="${speaker_id}-${gender}-${minutes}-${formatted_name}"
    
    local train_file="${filelist_dir}/${base_name}_audio_text_train_filelist.txt"
    local val_file="${filelist_dir}/${base_name}_audio_text_val_filelist.txt"
    local test_file="${filelist_dir}/${base_name}_audio_text_test_filelist.txt"

    if [ ! -f "$trans_file" ]; then
        echo "File $trans_file does not exist" >&2
        return
    fi

    echo "Processing transcription file: $trans_file"

    temp_file=$(mktemp)

    while IFS= read -r line; do
        local_file=$(echo "$line" | awk '{print $1}')
        text=$(echo "$line" | cut -d' ' -f2-)

        audio_file_wav="$dir_path/$local_file.wav"

        if [ -f "$audio_file_wav" ]; then
            relative_audio_file_wav=$(realpath --relative-to="$current_dir" "$audio_file_wav")
            echo "$relative_audio_file_wav|$text" >> "$temp_file"
        else
            echo "$audio_file_wav does not exist" >&2
        fi
    done < "$trans_file"
    
    total_lines=$(wc -l < "$temp_file")
    echo "Total lines: $total_lines"
    
    train_lines=$((total_lines * 80 / 100))
    val_lines=$((total_lines * 10 / 100))
    test_lines=$((total_lines - train_lines - val_lines))

    echo "Train lines: $train_lines"
    echo "Val lines: $val_lines"
    echo "Test lines: $test_lines"

    head -n "$train_lines" "$temp_file" >> "$train_file"
    sed -n "$((train_lines+1)), $((train_lines+val_lines))p" "$temp_file" >> "$val_file"
    tail -n "$test_lines" "$temp_file" >> "$test_file"

    rm "$temp_file"
}

while IFS='|' read -r id gender subset minutes name; do
    if [[ "$id" =~ ^\; ]] || [[ -z "$id" ]]; then
        continue
    fi

    id=$(echo "$id" | tr -d ' ')
    gender=$(echo "$gender" | tr -d ' ')
    subset=$(echo "$subset" | tr -d ' ')
    minutes=$(echo "$minutes" | tr -d ' ')
    name=$(echo "$name" | sed 's/^[ \t]*//;s/[ \t]*$//')

    echo "Processing speaker: ID=$id, Gender=$gender, Subset=$subset, Minutes=$minutes, Name=$name"

    find "$dataset_dir/$subset/$id" -name "*.trans.txt" | while read -r trans_file; do
        process_trans_file "$trans_file" "$subset" "$id" "$gender" "$minutes" "$name"
    done
done < "$speaker_info_file"

echo "Conversion completed. The filelists have been saved in $filelist_dir."