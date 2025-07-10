#!/bin/bash

# We'll need this to unzip our new toys!
# You might need to install it first!
# sudo yum install unzip

set -e

# Our NEW, super-special legacy model home!
DEST_DIR="/media/scratch/mferguson/legacy_models"
# A temporary place to put the downloads!
TEMP_DIR=$(mktemp -d)

echo "mkdir -p '$DEST_DIR'"
mkdir -p "$DEST_DIR"

echo "mkdir -p '$TEMP_DIR'"
mkdir -p "$TEMP_DIR"

# --- GloVe's Grand Entrance! ---
echo "⬇️  Here comes GloVe! It's a big one!"
kaggle datasets download -d anindya2906/glove6b -p "$TEMP_DIR"
unzip "$TEMP_DIR/glove6b.zip" -d "$DEST_DIR"
echo "✅ GloVe is home safe and sound!"

# --- Word2Vec's Wonderful Arrival! ---
echo "⬇️  And here's Word2Vec! ZOOOOOM!"
kaggle datasets download -d leadbest/googlenewsvectorsnegative300 -p "$TEMP_DIR"
unzip "$TEMP_DIR/googlenewsvectorsnegative300.zip" -d "$DEST_DIR"
echo "✅ Word2Vec has arrived!"

# Let's be tidy and clean up our temporary files!
rm -rf "$TEMP_DIR"

echo "🎉 YAY! All our legacy models are ready for action!"