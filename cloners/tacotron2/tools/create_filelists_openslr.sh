#!/bin/bash

output_file=$1
dataset_dir=$2

if [ -z "$output_file" ] || [ -z "$dataset_dir" ]; then
    echo "Usage: $0 <output_file> <dataset_dir>"
    exit 1
fi

> "$output_file"

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not installed"
    exit 1
fi

process_trans_file() {
    local trans_file=$1
    local dir_path=$(dirname "$trans_file")

    if [ ! -f "$trans_file" ]; then
        echo "File $trans_file does not exist" >&2
        return
    fi

    echo "Processing transcription file: $trans_file"

    awk -v dir_path="$dir_path" -v output_file="$output_file" '{
        local_file = $1
        $1 = ""
        text = substr($0, 2)
                
        audio_file_flac = dir_path "/" local_file ".flac"
        audio_file_wav = dir_path "/" local_file ".wav"

        if (system("[ -f \"" audio_file_flac "\" ]") == 0) {
            print "Converting " audio_file_flac " to " audio_file_wav
            if (system("ffmpeg -i \"" audio_file_flac "\" \"" audio_file_wav "\" -y -loglevel error") == 0) {
                print audio_file_wav "|" text >> output_file
            } else {
                print "Conversion failed for " audio_file_flac > "/dev/stderr"
            }       
        } else {
            print audio_file_flac " does not exist" > "/dev/stderr"
        }
    }' "$trans_file"
}

export -f process_trans_file
export output_file

find "$dataset_dir" -name "*.trans.txt" -exec bash -c 'process_trans_file "$0"' {} \;

echo "Conversion completed. The file has been saved as $output_file."