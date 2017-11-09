#!/usr/bin/env bash
## Text processing tools

# Convert from Latin-1 to UTF-8
iconv -f ISO-8859-15 -t UTF-8 $1 > out_utf.txt

# Remove colon, dots:
awk '{gsub(/\,|\./," ")}1' out_utf.txt > out_colons.txt
# Add spaces around hyphens
awk '{gsub(/\-/," - ")}1' out_colons.txt > out_hyphens.txt

# Lowercase for non-ascii charsets:
tr '[:upper:]' '[:lower:]' < out_hyphens.txt > out.txt



